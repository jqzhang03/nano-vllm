from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from conftest import _simulate_prefill
from nanovllm.config import Config
import os


def test_scheduler():
    seqs = [Sequence([0] * 1024) for _ in range(4)]
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(model=path, num_kvcache_blocks=64, kvcache_block_size=256)
    scheduler = Scheduler(config)
    bm = scheduler.block_manager
    _simulate_prefill(bm, seqs[0], seqs[0].num_tokens)
    seqs2 = [Sequence([0] * 1024) for _ in range(4)]
    for seq in seqs2:
        scheduler.add(seq)
    scheduled, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert len(scheduled) == 4
    for se in scheduled:
        assert se.num_scheduled_tokens == 1
    assert len(scheduler.waiting) == 0
    assert set(scheduled) == set(scheduler.running)
    
