import pytest
import torch

from nanovllm.layers.nvfp4_cublaslt import (
    nvfp4_scale_buffer_size,
    pad_nvfp4_scales,
    quantize_nvfp4_activations,
)
from nanovllm.layers.linear import (
    dequantize_nvfp4_tensor,
    nvfp4_gemm,
    quantize_nvfp4_tensor,
)


def test_nvfp4_e2m1_nibble_decode():
    low = torch.arange(8, dtype=torch.uint8)
    high = low | 0x8
    packed = (low | (high << 4)).unsqueeze(0)
    scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    tensor_scale = torch.ones((), dtype=torch.float32)

    actual = dequantize_nvfp4_tensor(packed, scale, tensor_scale, k=16)
    expected = torch.tensor([[0, -.0, .5, -.5, 1, -1, 1.5, -1.5,
                              2, -2, 3, -3, 4, -4, 6, -6]])
    assert torch.equal(actual, expected)


def test_nvfp4_quantize_pack_shapes_and_error():
    torch.manual_seed(11)
    weight = torch.randn(3, 80) * 0.2

    packed, block_scale, tensor_scale = quantize_nvfp4_tensor(weight)
    restored = dequantize_nvfp4_tensor(packed, block_scale, tensor_scale, k=80)

    assert packed.dtype == torch.uint8
    assert packed.shape == (3, 40)
    assert block_scale.dtype == torch.float8_e4m3fn
    assert block_scale.shape == (3, 5)
    assert tensor_scale.dtype == torch.float32 and tensor_scale.ndim == 0
    assert torch.isfinite(restored).all()
    assert (restored - weight).abs().mean() < weight.abs().mean() * 0.25


def test_nvfp4_rejects_nonstandard_k():
    with pytest.raises(ValueError, match="divisible by 16"):
        quantize_nvfp4_tensor(torch.randn(2, 30))


def test_cublaslt_scale_buffer_uses_tiled_k_panel_layout():
    scales = torch.full((2, 17), 0.5, dtype=torch.float8_e4m3fn)

    padded = pad_nvfp4_scales(scales)

    assert padded.shape == (nvfp4_scale_buffer_size(2, 17, operand="B"),)
    for row in range(2):
        for group in range(17):
            offset = (group // 4) * 512 + row * 16 + group % 4
            assert padded[offset].float() == scales[row, group].float()
    assert padded[4].float() == 1.0  # padding in the first four-scale panel
    assert padded[511].float() == 1.0


def test_config_accepts_nvfp4_mode(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from nanovllm.config import Config
    monkeypatch.setattr("nanovllm.config.AutoConfig.from_pretrained",
                        lambda _: SimpleNamespace(max_position_embeddings=512))
    config = Config(model=str(tmp_path), quantization="nvfp4")
    assert config.quantization == "nvfp4"
    native = Config(model=str(tmp_path), quantization="nvfp4", nvfp4_backend="cublaslt")
    assert native.nvfp4_backend == "cublaslt"


def test_cublaslt_quantization_stores_one_swizzled_scale_buffer():
    from nanovllm.layers.linear import ReplicatedLinear

    layer = ReplicatedLinear(128, 64)
    layer.weight.data.normal_()
    layer.quantize_nvfp4(backend="cublaslt")

    assert layer.w_nvfp4_scale.ndim == 1
    assert layer.w_nvfp4_scale.numel() == nvfp4_scale_buffer_size(64, 8, operand="B")
    assert not hasattr(layer, "w_nvfp4_scale_cublaslt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 GEMM requires CUDA")
def test_nvfp4_gemm_matches_dequantized_weight_reference():
    torch.manual_seed(23)
    m, n, k = 17, 67, 144  # partial M/N tiles and a K tail after the first 128 values
    activation = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed, block_scale, tensor_scale = quantize_nvfp4_tensor(weight)
    dequantized = dequantize_nvfp4_tensor(packed, block_scale, tensor_scale, k).to(weight.dtype)

    actual = nvfp4_gemm(activation, packed, block_scale, tensor_scale)
    expected = torch.nn.functional.linear(activation, dequantized)

    assert actual.shape == (m, n)
    assert torch.allclose(actual, expected, atol=0.1, rtol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 packing requires CUDA")
def test_nvfp4_activation_packing_round_trips_exact_fp4_values():
    pattern = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
        device="cuda", dtype=torch.bfloat16)
    activation = pattern.repeat(8).view(1, 128).repeat(3, 1)

    packed, scales = quantize_nvfp4_activations(activation)
    restored = dequantize_nvfp4_tensor(
        packed,
        torch.ones((3, 8), device="cuda", dtype=torch.float8_e4m3fn),
        torch.ones((), device="cuda"),
        k=128,
    )

    assert packed.shape == (3, 64)
    assert scales.shape == (nvfp4_scale_buffer_size(3, 8, operand="A"),)
    assert torch.equal(scales.float(), torch.ones_like(scales.float()))
    assert torch.equal(restored, activation.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 packing requires CUDA")
def test_cublaslt_activation_scales_use_tiled_k_panel_layout():
    m, groups = 3, 8
    activation = torch.empty((m, groups * 16), device="cuda", dtype=torch.bfloat16)
    for group in range(groups):
        activation[:, group * 16:(group + 1) * 16] = 6 * (group + 1)

    _, scales = quantize_nvfp4_activations(activation)

    assert scales.shape == (nvfp4_scale_buffer_size(m, groups, operand="A"),)
    for row in range(m):
        for group in range(groups):
            offset = (group // 4) * 512 + row * 16 + group % 4
            assert scales[offset].float() == float(group + 1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="native NVFP4 MMA requires a Blackwell-class GPU",
)
def test_cublaslt_nvfp4_gemm_matches_dequantized_reference():
    torch.manual_seed(29)
    m, n, k = 16, 64, 128
    pattern = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0],
        device="cuda", dtype=torch.bfloat16)
    activation = torch.stack(
        [torch.roll(pattern, shifts=row).repeat(k // 16) for row in range(m)])
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed, block_scale, tensor_scale = quantize_nvfp4_tensor(weight)
    dequantized = dequantize_nvfp4_tensor(packed, block_scale, tensor_scale, k)

    actual = nvfp4_gemm(
        activation, packed, pad_nvfp4_scales(block_scale), tensor_scale,
        backend="cublaslt")
    expected = (activation.float() @ dequantized.float().T).to(torch.bfloat16)

    assert actual.shape == (m, n)
    assert torch.allclose(actual.float(), expected.float(), atol=0.25, rtol=0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="native NVFP4 MMA requires a Blackwell-class GPU",
)
def test_cublaslt_nvfp4_gemm_accepts_empty_moe_expert_batch():
    k, n = 128, 64
    activation = torch.empty((0, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    packed, block_scale, tensor_scale = quantize_nvfp4_tensor(weight)

    actual = nvfp4_gemm(
        activation, packed, pad_nvfp4_scales(block_scale), tensor_scale,
        backend="cublaslt")

    assert actual.shape == (0, n)
    assert actual.dtype == torch.bfloat16
