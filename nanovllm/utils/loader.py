import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _resolve_tensor(model: nn.Module, packed_modules_mapping: dict, weight_name: str):
    """把 HF 张量名解析为 (param_name, shard_id)，与 eager 加载同一套 packed 映射。"""
    for k in packed_modules_mapping:
        if k in weight_name:
            v, shard_id = packed_modules_mapping[k]
            return weight_name.replace(k, v), shard_id
    return weight_name, None


def _chunk_of(weight_name: str) -> str:
    """张量所属的"加载块"：decoder 层整层一块，其余按父模块路径。

    流式加载按块物化+量化，块粒度 = 一个 decoder layer（7B 单层 fp16 ~0.5GB）。
    """
    parts = weight_name.split(".")
    if len(parts) >= 3 and parts[:2] == ["model", "layers"]:
        return ".".join(parts[:3])          # model.layers.N
    return ".".join(parts[:-1])             # 父模块路径，如 model.embed_tokens / model.norm / lm_head


def load_model(model: nn.Module, path: str, streaming: bool = False,
               chunk_hook=None) -> None:
    """加载权重到模型。

    streaming=False（默认）：一次性遍历所有 safetensors 文件加载（原行为）。

    streaming=True：**按层加载 + 加载即量化**（16GB 卡跑 7B+ 的前提，见 LEARNING.md
    阶段7）。模型须先在 meta 设备构造（不占显存）；这里按"顶层块"（embed_tokens /
    每个 decoder layer / norm / lm_head）逐个物化到 cuda → 加载权重 → 调用
    chunk_hook(module, chunk_path)（ModelRunner 用它立即量化该层，`del self.weight`
    释放 fp16），再处理下一块。chunk_path 是模块完整路径（如 "model.layers.0"），
    AWQ 缩放表按完整路径查找。任一时刻显存峰值 ≈ 累计量化权重 + 单个 fp16 层 + embed。
    绑定词表（tie_word_embeddings）的 lm_head 由调用方在加载后重新绑定
    （物化会打破 __init__ 里的存储共享）。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    if not streaming:
        _load_eager(model, path, packed_modules_mapping)
        return
    _load_streaming(model, path, packed_modules_mapping, chunk_hook)


def _load_eager(model: nn.Module, path: str, packed_modules_mapping: dict) -> None:
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name, shard_id = _resolve_tensor(model, packed_modules_mapping, weight_name)
                param = model.get_parameter(param_name)
                # 非 packed 参数（如 RMSNorm.weight）没有自定义 weight_loader → 默认拷贝
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is None:
                    weight_loader(param, f.get_tensor(weight_name))
                else:
                    weight_loader(param, f.get_tensor(weight_name), shard_id)


def _materialize(module: nn.Module, device: str) -> None:
    """meta → 真实设备物化。

    用 module.to_empty()（torch 文档要求；meta 参数不能 .to() 也不能 set_data）。
    代价：to_empty 用新 Parameter 对象替换旧对象，丢掉 __init__ 里挂到参数上的
    weight_loader 属性 → 物化后按模块重新挂回（加载仍按 packed 映射分片）。
    """
    module.to_empty(device=device)
    for m in module.modules():
        loader = getattr(m, "weight_loader", None)
        if loader is None:
            continue
        for p in m.parameters(recurse=False):
            if p is not None:
                p.weight_loader = loader


def _load_streaming(model: nn.Module, path: str, packed_modules_mapping: dict,
                    chunk_hook) -> None:
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert files, f"no .safetensors files in {path}"
    # pass 1：索引所有张量 → 所属块（safetensors 的张量不会跨分片，块内张量可能分散在多个文件）
    chunks: dict[str, list] = {}
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                chunks.setdefault(_chunk_of(weight_name), []).append(
                    (_resolve_tensor(model, packed_modules_mapping, weight_name), weight_name, file))
    # 块顺序：embed_tokens → layers 0..N-1 → 其余（norm / lm_head 等，按名字排序保持稳定）
    def sort_key(chunk: str):
        parts = chunk.split(".")
        if len(parts) == 3 and parts[0] == "model" and parts[1] == "layers":
            return (0, int(parts[2]), chunk)
        if chunk == "model.embed_tokens":
            return (-1, 0, chunk)
        return (1, 0, chunk)
    for chunk in sorted(chunks, key=sort_key):
        chunk_module = model.get_submodule(chunk)
        # 物化：meta → cuda（保留 dtype/shape，未初始化），随后由加载填充。
        # 手动物化（保留 weight_loader）；torch 2.8 也禁止对 meta 模块用 .to()
        if any(p.is_meta for p in chunk_module.parameters()):
            _materialize(chunk_module, "cuda")
        for (param_name, shard_id), weight_name, file in chunks[chunk]:
            param = model.get_parameter(param_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            with safe_open(file, "pt", "cpu") as f:
                if shard_id is None:
                    weight_loader(param, f.get_tensor(weight_name))
                else:
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
        if chunk_hook is not None:
            chunk_hook(chunk_module, chunk)
