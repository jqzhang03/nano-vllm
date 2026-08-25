"""模型注册表 + Gemma-2 layer_types 解析 + Attention SWA/softcap 参数的纯 Python 单测。"""
import pytest


def test_registry_mistral_gemma2():
    from nanovllm.models.registry import get_model_class, _MODEL_REGISTRY
    assert "mistral" in _MODEL_REGISTRY
    assert "gemma2" in _MODEL_REGISTRY
    cls = get_model_class("mistral")
    assert cls.__name__ == "MistralForCausalLM"
    cls = get_model_class("gemma2")
    assert cls.__name__ == "Gemma2ForCausalLM"


def test_registry_unknown_and_planned():
    from nanovllm.models.registry import get_model_class
    with pytest.raises(ValueError):
        get_model_class("definitely-not-a-model")
    with pytest.raises(NotImplementedError):
        get_model_class("mixtral")


def test_gemma2_layer_types_9b_pattern():
    """9B 式 attention_types=[[["global","local"],5]] → 前10层交替（归一化 local→sliding），其余 sliding。"""
    from nanovllm.models.gemma2 import gemma2_layer_types

    class Cfg:
        attention_types = [[["global", "local"], 5]]

    types = gemma2_layer_types(Cfg, 42)
    assert len(types) == 42
    assert types[:10] == ["global", "sliding"] * 5
    assert types[10:] == ["sliding"] * 32  # 超出模式长度取最后一项


def test_gemma2_layer_types_sliding_attention_alias():
    """新版 config 用 "sliding_attention" 命名 → 归一化为 "sliding"。"""
    from nanovllm.models.gemma2 import gemma2_layer_types

    class Cfg:
        attention_types = [[["sliding_attention", "global"], 1]]

    types = gemma2_layer_types(Cfg, 2)
    assert types == ["sliding", "global"]


def test_gemma2_layer_types_2b_missing():
    """2B/9B-it（unsloth 转换版）无 layer_types/attention_types → 复刻安装版 transformers
    默认：sliding 开头交替（实证 13/13、21/21）。"""
    from nanovllm.models.gemma2 import gemma2_layer_types

    class Cfg:
        pass

    types = gemma2_layer_types(Cfg, 26)
    assert types == ["sliding" if (i + 1) % 2 else "global" for i in range(26)]
    assert types[:4] == ["sliding", "global", "sliding", "global"]


def test_gemma2_layer_types_explicit():
    """新 transformers 的显式 layer_types 字段直接使用 + 归一化。"""
    from nanovllm.models.gemma2 import gemma2_layer_types

    class Cfg:
        layer_types = ["sliding_attention", "full_attention", "sliding_attention"]

    types = gemma2_layer_types(Cfg, 4)
    assert types == ["sliding", "global", "sliding", "sliding"]  # 不足补齐取末项


def test_attention_window_flash_params():
    """Attention 的 window_size → flash window_size 参数映射（probe 钉死的约定）。"""
    from nanovllm.layers.attention import Attention
    a = Attention(8, 128, 128 ** -0.5, 4, window_size=4096)
    assert a._flash_window == (4095, 0)      # flash [i-left, i] 含两端 → W-1
    assert a._flash_softcap == 0.0
    a2 = Attention(8, 128, 128 ** -0.5, 4, window_size=None, logit_softcapping=50.0)
    assert a2._flash_window == (-1, -1)
    assert a2._flash_softcap == 50.0


def test_attention_softcap_fp8_conflict():
    """softcap 层不允许 fp8 KV（自研 fp8 内核无 softcap）——forward 断言。"""
    from nanovllm.layers.attention import Attention
    a = Attention(8, 128, 128 ** -0.5, 4, logit_softcapping=50.0)
    a.use_fp8 = True
    with pytest.raises(AssertionError):
        a.forward(None, None, None)


def test_mistral_gemma2_construct_cpu():
    """小 config 的模型构造（CPU，无 forward）：结构字段正确。"""
    from transformers import MistralConfig, Gemma2Config
    from nanovllm.models.mistral import MistralForCausalLM
    from nanovllm.models.gemma2 import Gemma2ForCausalLM

    mc = MistralConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, intermediate_size=128, vocab_size=100,
                       max_position_embeddings=512, sliding_window=256)
    m = MistralForCausalLM(mc)
    assert m.model.layers[0].self_attn.attn.window_size == 256
    assert m.model.layers[0].self_attn.attn._flash_window == (255, 0)

    gc = Gemma2Config(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=32, intermediate_size=128,
                      vocab_size=100, max_position_embeddings=512,
                      layer_types=["sliding_attention", "full_attention",
                                   "sliding_attention", "full_attention"],
                      attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                      sliding_window=256, query_pre_attn_scalar=32)
    g = Gemma2ForCausalLM(gc)
    attns = [g.model.layers[i].self_attn.attn for i in range(4)]
    assert [a.window_size for a in attns] == [256, None, 256, None]  # 交替 local/global
    assert [a.logit_softcapping for a in attns] == [50.0] * 4
    assert g.config.final_logit_softcapping == 30.0
    # gate/up 独立模块（非 merged）→ packed 映射只含 qkv
    assert set(g.packed_modules_mapping) == {"q_proj", "k_proj", "v_proj"}
