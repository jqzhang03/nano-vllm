import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化NCCL分布式进程组，所有GPU通过localhost:2333进行通信，用于张量并行时的命令执行
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 根据配置初始化模型结构
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载预训练权重到模型上
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # W8A8权重量化（SmoothQuant式：校准激活通道分布→折叠进权重→per-channel int8 + Triton int8 GEMM）
        # 放在sampler之后：校准用的run()会调用sampler
        if config.quantization == "w8a8":
            self.calibrate_and_quantize_w8a8()
        # 调用预热方法，执行一次模拟prefill来分配显存、初始化CUDA内核，并测量峰值显存
        self.warmup_model()
        # 分配KV Cache的显存空间，并根据模型层数将KV Cache引用绑定到各注意力层
        self.allocate_kv_cache()
        # 若没有强制eager模式，则使用CUDA图优化加速decode阶段
        if not self.enforce_eager:
            self.capture_cudagraph()
            # 投机解码：再捕获verify前向（varlen）的graph族（固定容量+空行填充）
            if config.speculative == "ngram":
                self.capture_spec_graph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # 如果启用了张量并行，rank为0的进程为主进程，创建1MB共享内存用于向worker发布任务
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 所有进程同步，确保主进程创建完了再访问
                dist.barrier()
            else:
                dist.barrier()
                # 打开共享内存
                self.shm = SharedMemory(name="nanovllm")
                # 进程进入无限循环，读取共享内存中的任务并执行，直到收到exit信号
                self.loop()

    def exit(self):

        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # 删除存储的CUDA Graph对象和内存池，释放显存
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, "spec_graphs"):
                del self.spec_graphs, self.spec_graph_pool, self.spec_graph_vars
        torch.cuda.synchronize()
        # 销毁分布式进程组，释放NCCL资源
        dist.destroy_process_group()

    def loop(self):
        # 无限循环执行任务，直到收到exit命令
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 等待主进程通过Event通知有新数据可读
        self.event.wait()
        # 读取共享内存前4字节，得到序列话数据的长度，小端序
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 从共享内存偏移4字节处开始读n个字节，利用pickle反序列化出方法名和参数
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清理Event状态，等待下一次通知
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        # 将方法名和参数序列化为字节
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 将长度n以小端序写入共享内存前4字节
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            # 设置每个Event，通知所有worker进程读取任务
            event.set()

    # 统一的远程调用入口
    def call(self, method_name, *args):
        # 如果是张量并行且是主进程，先写入共享内存，再调用
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 根据方法名字符串动态获取对应的方法对象
        method = getattr(self, method_name, None)
        return method(*args)

    def cow_block(self, old_block_id: int, new_block_id: int):
        """COW：把旧块的KV内容复制到新块。

        kv_cache布局为 [2(K/V), num_layers, num_blocks, block_size, num_kv_heads, head_dim]，
        按块下标切片即一次全层K/V复制（设备到设备，无需自定义内核）。
        TP>1时每个rank复制自己的KV分片，块表与COW对全rank一致，天然对齐。
        """
        self.kv_cache[:, :, new_block_id] = self.kv_cache[:, :, old_block_id]

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 计算单序列最大长度
        seq_len = min(max_num_batched_tokens, max_model_len)
        # 计算预热使用的序列数
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, "prefill")
        torch.cuda.empty_cache()
        if self.config.kv_cache_dtype == "fp8_e4m3":
            # FP8 KV缓存：用随机token的prefill前向校准每层K/V的固定scale
            # （scale = max|·| / 448 * 安全系数；量化误差受E4M3的3位尾数限制）
            self.calibrate_fp8_kv()

    def calibrate_and_quantize_w8a8(self):
        """校准每个线性层输入的逐通道激活范围（SmoothQuant的s向量），然后折叠进权重并量化。"""
        from nanovllm.layers.linear import LinearBase
        linears = [m for m in self.model.modules() if isinstance(m, LinearBase)]
        hooks = []
        for m in linears:
            m.x_max = None  # 临时属性：校准期间累计本层输入逐通道amax
            def make_hook(mod):
                def hook(_mod, args):
                    x = args[0].float()
                    amax = x.abs().amax(dim=0)
                    if mod.x_max is None:
                        mod.x_max = amax.cpu()
                    else:
                        mod.x_max = torch.maximum(mod.x_max, amax.cpu())
                return hook
            hooks.append(m.register_forward_pre_hook(make_hook(m)))
        # 用随机token的prefill前向校准（与FP8 KV校准同源数据分布）
        seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        vocab = min(self.config.hf_config.vocab_size, 50000)
        rng = torch.Generator(device="cuda").manual_seed(42)
        seqs = [Sequence(torch.randint(0, vocab, (seq_len,), generator=rng).tolist()) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, "prefill")
        for h in hooks:
            h.remove()
        for m in linears:
            m.quantize_w8a8(m.x_max)
            del m.x_max

    def calibrate_fp8_kv(self):
        layers = [m for m in self.model.modules() if hasattr(m, "calibrating")]
        assert layers, "no attention layers found for FP8 KV calibration"
        seq_len = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        num_seqs = min(self.config.max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 用常用token范围（id < 50000）校准：全词表随机会引入稀有token的激活异常值，
        # 把scale撑大导致典型值量化变粗
        vocab = min(self.config.hf_config.vocab_size, 50000)
        rng = torch.Generator(device="cuda").manual_seed(42)
        seqs = [Sequence(torch.randint(0, vocab, (seq_len,), generator=rng).tolist()) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        for m in layers:
            m.calibrating = True
        self.run(seqs, "prefill")
        for m in layers:
            m.calibrating = False
            m.k_scale = max(m.cal_max_k, 1e-6) / 448.0 * 1.1
            m.v_scale = max(m.cal_max_v, 1e-6) / 448.0 * 1.1
            m.inv_k_scale = 1.0 / m.k_scale
            m.inv_v_scale = 1.0 / m.v_scale

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 查看当前GPU的空闲显存和总显存
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # 获取预热阶段达到的Pytorch分配显存峰值，不包括caching allocator的保留池
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # 获取当前Pytorch已分配的显存
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 张量并行后每个GPU上的KV头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 获取每个注意力头的维度，优先取head_dim，否则用隐藏层注意力维度处以注意力头数
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # FP8 KV缓存：E4M3每元素1字节，块大小减半 → 同样显存下块数翻倍
        use_fp8 = config.kv_cache_dtype == "fp8_e4m3"
        itemsize = 1 if use_fp8 else hf_config.dtype.itemsize
        # 单KV Cache块大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * itemsize
        # 根据可用显存计算可分配的KV Cache块总数
        # 总显存x利用率-已使用显存-预热峰值+当前已分配了的显存 除以 单块大小
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        kv_dtype = torch.float8_e4m3fn if use_fp8 else hf_config.dtype
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
                                    self.block_size, num_kv_heads, head_dim, dtype=kv_dtype)
        layer_id = 0
        for module in self.model.modules():
            # 如果模块是注意力层，将KV Cache赋值给该层的KV Cache，模型层数+1
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                module.use_fp8 = use_fp8
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        # 找到序列中最长的块表长度
        max_len = max(len(seq.block_table) for seq in seqs)
        # 将不足最长块表长度的通过添加-1补足长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # 转为int32张量，使用锁页内存加速传输，并异步拷贝到GPU
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """纯prefill批次（含前缀缓存读取）。返回 (input_ids, positions)。"""
        input_ids = [] # 所有序列本次prefill的输入token ID
        positions = [] # 对应的位置索引
        cu_seqlens_q = [0] # query的累积序列长度，用于flash attn的变长输入
        cu_seqlens_k = [0] # key的累积序列长度
        max_seqlen_q = 0 # query最大序列长度
        max_seqlen_k = 0 # key最大序列长度
        slot_mapping = [] # 每个token应写入KV Cache的槽位索引
        block_tables = None # 块表，如果key长度大于query长度(使用了前缀缓存)则需要构建
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup没有块表，跳过slot_mapping生成
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        # key的总序列大于query的总序列长度，有prefix cache，query跳过了前缀，则需要准备块表供注意力内核使用
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        # 将数据转成int64张量，锁页传输并异步拷贝到GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 将prefill上下文设置到全局上下文，供注意力算子使用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_mixed(self, seqs: list[Sequence]):
        """混合批次（prefill行在前 + decode行在后，vLLM V1同款调度）。

        Context语义：cu_seqlens_q/k、max_seqlen_q/k 为 prefill 组的量
        （flash varlen 只跑 prefill 行）；slot_mapping 覆盖全批次（写入）；
        context_lens/block_tables 为 decode 组（供 decode 注意力内核）。
        """
        if any(seq.draft_tokens is not None for seq in seqs):
            # 投机混合步：verify行不是flash-kvcache行，走全批次varlen
            return self._prepare_mixed_spec(seqs)
        # --- prefill 部分（前 n_prefill 个seq） ---
        n_prefill = sum(1 for seq in seqs if seq.is_prefill)
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        prefill_block_tables = None
        for seq in seqs[:n_prefill]:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                slot_end = (seq.block_table[i] * self.block_size + self.block_size
                            if i != end_block - 1
                            else seq.block_table[i] * self.block_size + end - i * self.block_size)
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            prefill_block_tables = self.prepare_block_tables(seqs[:n_prefill])
        n_prefill_tokens = cu_seqlens_q[-1]
        # --- decode 部分（后 len-n_prefill 个seq） ---
        decode_seqs = seqs[n_prefill:]
        context_lens = []
        for seq in decode_seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        block_tables = self.prepare_block_tables(decode_seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, context_lens, block_tables,
                    is_mixed=True, prefill_block_tables=prefill_block_tables,
                    n_prefill_tokens=n_prefill_tokens)
        return input_ids, positions

    def prepare_spec(self, seqs: list[Sequence]):
        """纯verify步（投机）：每序列query = [末token, 草稿...]（γ+1个token，
        位置从 len-1 起），num_cached = len-1 —— 即"分块prefill + 前缀复用"：
        KV恒来自缓存（cache形状 + block_tables），logits保留全部行供验收。

        logits语义：位置 len-1+i 的logit预测位置 len+i → 样本s_i验证草稿d_i，
        最后一行（位置 len+γ-1）是全接受时的bonus。"""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for seq in seqs:
            start = len(seq) - 1
            n = seq.num_scheduled_tokens
            end = start + n
            assert n == len(seq.draft_tokens) + 1
            input_ids.extend([seq.last_token] + list(seq.draft_tokens))
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + n)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(n, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            assert seq.block_table
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                slot_end = (seq.block_table[i] * self.block_size + self.block_size
                            if i != end_block - 1
                            else seq.block_table[i] * self.block_size + end - i * self.block_size)
                slot_mapping.extend(range(slot_start, slot_end))
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables,
                    is_spec=True)
        return input_ids, positions

    def _prepare_mixed_spec(self, seqs: list[Sequence]):
        """投机混合步：prefill行（query = 剩余prompt chunk） + verify行
        （query = [末token, 草稿]）全部走varlen；verify行恒有前缀复用 →
        KV恒为缓存形状（flash按k.shape[1]推断block size），block_tables覆盖全批次。

        Context：cu_seqlens_q/k 覆盖全批次；n_prefill_rows = prefill组行数
        （LM head 对 prefill 组取末行、verify 组保留全行）。"""
        n_prefill_rows = sum(1 for seq in seqs if seq.is_prefill)
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for seq in seqs:
            if seq.is_prefill:
                start = seq.num_cached_tokens
                n = seq.num_scheduled_tokens
                tokens = seq[start:start + n]
            else:
                start = len(seq) - 1
                n = seq.num_scheduled_tokens
                assert n == len(seq.draft_tokens) + 1
                tokens = [seq.last_token] + list(seq.draft_tokens)
            end = start + n
            input_ids.extend(tokens)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + n)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(n, max_seqlen_q)
            max_seqlen_k = max(end, max_seqlen_k)
            assert seq.block_table
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                slot_end = (seq.block_table[i] * self.block_size + self.block_size
                            if i != end_block - 1
                            else seq.block_table[i] * self.block_size + end - i * self.block_size)
                slot_mapping.extend(range(slot_start, slot_end))
        block_tables = self.prepare_block_tables(seqs)
        n_prefill_tokens = cu_seqlens_q[n_prefill_rows]

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables,
                    is_mixed=True, prefill_block_tables=block_tables,
                    n_prefill_tokens=n_prefill_tokens, is_spec=True, n_prefill_rows=n_prefill_rows)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 温度按logits行数展开：普通行每seq 1行；verify行每seq γ+1行（LM head保留全行）
        temperatures = []
        for seq in seqs:
            n = 1 if seq.draft_tokens is None else seq.num_scheduled_tokens
            temperatures.extend([seq.temperature] * n)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # 装饰器，该方法在推理模式下使用，禁用梯度计算和额外的梯度计算性能优化
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, kind: str):
        # 纯spec批次（verify）走spec CUDA graph：固定容量+零长度行填充，消除eager的
        # 逐kernel启动税（实测~10ms/步，见BENCHMARKS.md §9）。按本步最大query长度
        # 选stride家族（低γ步用stride-3图，容量=3×行数，减少填充浪费）；fp8的verify
        # 路径（每层全缓存反量化）不吃graph收益，保持eager。
        if (kind == "spec" and not self.enforce_eager
                and self.config.kv_cache_dtype != "fp8_e4m3"
                and getattr(self, "spec_graphs", None)):
            context = get_context()
            rows = context.cu_seqlens_q.size(0) - 1
            stride = context.max_seqlen_q
            s = next((x for x in self.spec_strides if x >= stride), None)
            cap = next((r for r in self.spec_graph_rows if r >= rows), None)
            if s is not None and cap is not None and (s, cap) in self.spec_graphs:
                return self._run_spec_graph(input_ids, positions, s, cap, rows)
        # 只有纯decode批次且非强制eager且batch<=512时走CUDA graph；
        # prefill与mixed批次一律eager（mixed含prefill行，无法用纯decode图）
        if kind != "decode" or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else: # 否则执行decode
            # 获取batch_size
            bs = input_ids.size(0)
            # 获取全局上下文
            context = get_context()
            # 复用固定batch_size的图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            # 获取图中使用的静态变量
            graph_vars = self.graph_vars
            # 将当前batch的input_ids拷贝至图张量的前bs个位置
            graph_vars["input_ids"][:bs] = input_ids
            # 将当前batch的positions拷贝至图张量的前bs个位置
            graph_vars["positions"][:bs] = positions
            # 将slot_mapping全部赋值为-1
            graph_vars["slot_mapping"].fill_(-1)
            # 将前batch_size个位置填入有效slot_mapping
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # 将context_lens清零
            graph_vars["context_lens"].zero_()
            # 将前batch_size个context_lens赋值
            graph_vars["context_lens"][:bs] = context.context_lens
            # 将block_tables拷贝至图张量相应位置
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # 重放CUDA Graph，执行模型前向
            graph.replay()
            # 从图输出张量中取出前batch_size个隐藏状态，计算完logits后返回
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _run_spec_graph(self, input_ids: torch.Tensor, positions: torch.Tensor,
                        stride: int, cap_rows: int, rows: int) -> torch.Tensor:
        """重放verify CUDA graph：真实行拷入静态张量，尾部用零长度行填充。

        - cu_seqlens 尾部重复末值 → flash varlen 按空行跳过（bit-exact，probe验证）；
        - slot_mapping 填充 -1（store_kvcache跳过）；input_ids/positions 填 0；
        - LM head 在图外对真实行切片计算（省掉填充行的词表GEMM）。
        """
        config = self.config
        context = get_context()
        cap_tok = cap_rows * stride
        real_tok = input_ids.size(0)
        v = self.spec_graph_vars
        v["input_ids"][:real_tok] = input_ids
        v["input_ids"][real_tok:cap_tok] = 0
        v["positions"][:real_tok] = positions
        v["positions"][real_tok:cap_tok] = 0
        v["slot_mapping"][:real_tok] = context.slot_mapping
        v["slot_mapping"][real_tok:cap_tok] = -1
        v["cu_seqlens_q"][:rows + 1] = context.cu_seqlens_q
        v["cu_seqlens_q"][rows + 1:cap_rows + 1] = context.cu_seqlens_q[-1]
        v["cu_seqlens_k"][:rows + 1] = context.cu_seqlens_k
        v["cu_seqlens_k"][rows + 1:cap_rows + 1] = context.cu_seqlens_k[-1]
        v["block_tables"][:rows, :context.block_tables.size(1)] = context.block_tables
        v["block_tables"][rows:cap_rows] = 0
        self.spec_graphs[(stride, cap_rows)].replay()
        return self.model.compute_logits(v["outputs"][:real_tok])

    @torch.inference_mode()
    def capture_spec_graph(self):
        """捕获verify前向（varlen）的CUDA graph族（投机解码专用）。

        verify步的行数每步都变 → 按行容量捕获一组图；重放时真实行 + 零长度填充行。
        捕获时全行满长度：flash varlen 的 grid 按容量烘焙，重放的真实 cu_seqlens
        只是数据（bit-exact，benchmarks/_graph_pad_probe.py 验证）。max_seqlen_q
        (=stride) 与 max_seqlen_k (=max_model_len) 作为标量参数在捕获时固定——
        flash varlen 的 key 循环由 cu_seqlens 驱动，烘焙上限不引入开销（probe验证）。

        双stride家族：stride=γ_max+1 覆盖任意行；stride=3 覆盖低γ步（容量=3×行数，
        填充浪费减半——低γworkload的填充成本会超过CPU税节省）。容量上限256行
        （stride5下1280 token）受warmup峰值/显存约束；更大的verify步回落eager。
        LM head在图外（_run_spec_graph只对真实行切片计算）。
        """
        config = self.config
        hf_config = config.hf_config
        max_rows = min(self.config.max_num_seqs, 512, 256)
        self.spec_graph_rows = [r for r in (8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256)
                                if r <= max_rows] or [max_rows]
        self.spec_strides = sorted({config.max_draft_len + 1, 3})
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_stride = max(self.spec_strides)
        max_tok = max_rows * max_stride
        input_ids = torch.zeros(max_tok, dtype=torch.int64)
        positions = torch.zeros(max_tok, dtype=torch.int64)
        slot_mapping = torch.zeros(max_tok, dtype=torch.int32)
        cu_seqlens_q = torch.zeros(max_rows + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_rows + 1, dtype=torch.int32)
        block_tables = torch.zeros(max_rows, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_tok, hf_config.hidden_size)
        self.spec_graphs = {}
        self.spec_graph_pool = None

        for stride in self.spec_strides:
            for cap_rows in reversed(self.spec_graph_rows):
                cap_tok = cap_rows * stride
                # 捕获时全行满长度（grid按容量烘焙；重放时真实cu_seqlens只是数据）
                cu_seqlens_q[:cap_rows + 1] = torch.arange(0, cap_tok + 1, stride, dtype=torch.int32)
                cu_seqlens_k[:cap_rows + 1] = cu_seqlens_q[:cap_rows + 1]
                set_context(True, cu_seqlens_q[:cap_rows + 1], cu_seqlens_k[:cap_rows + 1],
                            stride, config.max_model_len, slot_mapping[:cap_tok], None,
                            block_tables[:cap_rows], is_spec=True)
                graph = torch.cuda.CUDAGraph()
                outputs[:cap_tok] = self.model(input_ids[:cap_tok], positions[:cap_tok])  # warmup（含编译）
                with torch.cuda.graph(graph, self.spec_graph_pool):
                    outputs[:cap_tok] = self.model(input_ids[:cap_tok], positions[:cap_tok])  # capture
                if self.spec_graph_pool is None:
                    self.spec_graph_pool = graph.pool()
                self.spec_graphs[(stride, cap_rows)] = graph
                torch.cuda.synchronize()
                reset_context()

        self.spec_graph_vars = dict(input_ids=input_ids, positions=positions,
                                    slot_mapping=slot_mapping, cu_seqlens_q=cu_seqlens_q,
                                    cu_seqlens_k=cu_seqlens_k, block_tables=block_tables,
                                    outputs=outputs)

    # run流程：准备输入、运行模型、采样、返回生成的token id列表
    def run(self, seqs: list[Sequence], kind: str, return_logits: bool = False):
        if kind == "mixed":
            input_ids, positions = self.prepare_mixed(seqs)
        elif kind == "prefill":
            input_ids, positions = self.prepare_prefill(seqs)
        elif kind == "spec":
            input_ids, positions = self.prepare_spec(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, kind)
        # 主进程通过采样器从logits中采样得到token id列表
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        if return_logits:
            # 精度检查用：返回本步logits（fp32，driver侧）
            return token_ids, logits.float() if self.rank == 0 else None
        return token_ids

    # 捕获CUDA Graph以加速decode阶段
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 最大batch_size不超过512
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 定义需要捕获的batch_size列表，初始时内存池为空
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小遍历batch_size，有利于内存池复用
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置decode阶段对应batch_size的上下文
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 运行一次作为warmup，分配所需显存并初始化静态缓冲
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # CUDA Graph捕获，若已有pool则复用
            with torch.cuda.graph(graph, self.graph_pool):
                # 在捕获的Graph中执行模型前向
                # 在捕获阶段，GPU不进行任何该阶段下的工作
                # 在CPU中，每次torch操作变成一次kernel launch API调用，driver在调用栈里拦截它并记录成一个节点，然后返回。CPU不等待、不调度、不执行
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            # 如果是第一次捕获，则记录该graph的内存池，后续graph共享此内存池以减少显存碎片
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            # 存储捕获的graph
            self.graphs[bs] = graph
            # 同步CUDA，保证捕获完成，从warmup到capture阶段，只有warmup在GPU中执行，因为warmup是异步的，因此需要等待warmup完成
            # 确保GPU完全跑空、分配器状态稳定，保证每一轮捕获都是从干净的边界开始，否则在捕获时可能会混入warmup阶段中的操作
            torch.cuda.synchronize()
            reset_context()

        # 保存静态张量字典，后续run_model时通过更新这些张量来改变graph的输入和输出
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
