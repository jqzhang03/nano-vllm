"""Native NVFP4 GEMM through cuBLASLt block-scaled matmul."""

from __future__ import annotations

import ctypes
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


_GROUP_SIZE = 16
_MAX_FP4 = 6.0
_MAX_E4M3 = 448.0
_MIN_E4M3 = 1.0 / 512.0
_GROUPS_PER_PROGRAM = 4
_SCALE_GROUPS_PER_PANEL = 4
_SCALE_TILE_BYTES = 512
_SCALE_A_TILE_ROWS = 32
_SCALE_B_TILE_ROWS = 128
_MAX_CACHED_PLANS = 64


@triton.jit
def _pack_nvfp4_activations_kernel(
    a_ptr,
    packed_ptr,
    scale_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    K_GROUPS: tl.constexpr,
    K_PANELS: tl.constexpr,
    GROUPS_PER_PROGRAM: tl.constexpr,
    SCALE_GROUPS_PER_PANEL: tl.constexpr,
    SCALE_TILE_BYTES: tl.constexpr,
    SCALE_A_TILE_ROWS: tl.constexpr,
):
    """Pack activations and write cuBLASLt's tiled per-16 scale layout."""
    row = tl.program_id(0)
    groups = (tl.program_id(1) * GROUPS_PER_PROGRAM
              + tl.arange(0, GROUPS_PER_PROGRAM))
    values = tl.arange(0, 16)
    k = groups[:, None] * 16 + values[None, :]
    valid_group = groups < K_GROUPS
    x = tl.load(
        a_ptr + row * stride_am + k * stride_ak,
        mask=valid_group[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=1)
    raw_scale = tl.where(amax > 0.0, amax / 6.0, 1.0)
    raw_scale = tl.minimum(tl.maximum(raw_scale, 1.0 / 512.0), 448.0)
    scale_fp8 = raw_scale.to(tl.float8e4nv)
    scale = scale_fp8.to(tl.float32)
    normalized = tl.abs(x) / scale[:, None]

    magnitude = (normalized > 0.25).to(tl.uint8)
    magnitude += (normalized > 0.75).to(tl.uint8)
    magnitude += (normalized > 1.25).to(tl.uint8)
    magnitude += (normalized > 1.75).to(tl.uint8)
    magnitude += (normalized > 2.5).to(tl.uint8)
    magnitude += (normalized > 3.5).to(tl.uint8)
    magnitude += (normalized > 5.0).to(tl.uint8)
    sign = tl.where(x < 0.0, 8, 0).to(tl.uint8)
    code = magnitude | sign

    pairs = tl.arange(0, 8)
    group_lane = tl.arange(0, GROUPS_PER_PROGRAM)
    even_idx = group_lane[:, None] * 0 + (pairs * 2)[None, :]
    odd_idx = group_lane[:, None] * 0 + (pairs * 2 + 1)[None, :]
    low = tl.gather(code, even_idx, axis=1)
    high = tl.gather(code, odd_idx, axis=1)
    packed = low | (high << 4)

    tl.store(
        packed_ptr + row * (K // 2) + groups[:, None] * 8
        + pairs[None, :],
        packed,
        mask=valid_group[:, None],
    )
    scale_value = tl.where(valid_group, scale, 1.0)
    k_panel = groups // SCALE_GROUPS_PER_PANEL
    scale_offset = (
        (row // SCALE_A_TILE_ROWS) * K_PANELS + k_panel
    ) * SCALE_TILE_BYTES
    scale_offset += (row % SCALE_A_TILE_ROWS) * 16
    scale_offset += groups % SCALE_GROUPS_PER_PANEL
    tl.store(
        scale_ptr + scale_offset,
        scale_value,
        mask=valid_group,
    )


@triton.jit
def _swizzle_nvfp4_weight_scales_kernel(
    source_ptr,
    destination_ptr,
    ROWS,
    GROUPS,
    K_PANELS: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_GROUPS_PER_PANEL: tl.constexpr,
    SCALE_TILE_BYTES: tl.constexpr,
    SCALE_B_TILE_ROWS: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < ROWS * GROUPS
    row = index // GROUPS
    group = index % GROUPS
    k_panel = group // SCALE_GROUPS_PER_PANEL
    n_tile = row // SCALE_B_TILE_ROWS
    n_in_tile = row % SCALE_B_TILE_ROWS
    offset = (n_tile * K_PANELS + k_panel) * SCALE_TILE_BYTES
    offset += (n_in_tile % 32) * 16
    offset += (n_in_tile // 32) * SCALE_GROUPS_PER_PANEL
    offset += group % SCALE_GROUPS_PER_PANEL
    value = tl.load(source_ptr + index, mask=mask, other=1.0)
    tl.store(destination_ptr + offset, value, mask=mask)


def nvfp4_scale_buffer_size(rows: int, groups: int, *, operand: str) -> int:
    """Return cuBLASLt's packed scale-buffer size for one matrix operand."""
    if rows < 0 or groups < 0:
        raise ValueError("scale dimensions must be non-negative")
    k_panels = triton.cdiv(groups, _SCALE_GROUPS_PER_PANEL)
    if operand == "A":
        row_tiles = triton.cdiv(rows, _SCALE_A_TILE_ROWS)
    elif operand == "B":
        row_tiles = triton.cdiv(rows, _SCALE_B_TILE_ROWS)
    else:
        raise ValueError(f"operand must be 'A' or 'B', got {operand!r}")
    return row_tiles * k_panels * _SCALE_TILE_BYTES


def pad_nvfp4_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack logical ``[N, K/16]`` E4M3 scales into cuBLASLt's SFB layout.

    cuBLASLt groups four K scales into 512-byte panels. Within a panel, output
    rows are tiled in groups of 128 and the four scale values are interleaved
    into 16-byte row slots.
    """
    if scales.ndim != 2 or scales.dtype != torch.float8_e4m3fn:
        raise ValueError("NVFP4 block scales must be a 2D E4M3 tensor")
    rows, groups = scales.shape
    k_panels = triton.cdiv(groups, _SCALE_GROUPS_PER_PANEL)
    packed = torch.ones(
        nvfp4_scale_buffer_size(rows, groups, operand="B"),
        device=scales.device,
        dtype=scales.dtype,
    )
    if rows == 0 or groups == 0:
        return packed
    source = scales.contiguous()
    if scales.is_cuda:
        _swizzle_nvfp4_weight_scales_kernel[
            (triton.cdiv(rows * groups, 256),)
        ](
            source,
            packed,
            rows,
            groups,
            K_PANELS=k_panels,
            BLOCK=256,
            SCALE_GROUPS_PER_PANEL=_SCALE_GROUPS_PER_PANEL,
            SCALE_TILE_BYTES=_SCALE_TILE_BYTES,
            SCALE_B_TILE_ROWS=_SCALE_B_TILE_ROWS,
            num_warps=4,
        )
    else:
        for row in range(rows):
            n_tile = row // _SCALE_B_TILE_ROWS
            n_in_tile = row % _SCALE_B_TILE_ROWS
            for group in range(groups):
                k_panel = group // _SCALE_GROUPS_PER_PANEL
                offset = (n_tile * k_panels + k_panel) * _SCALE_TILE_BYTES
                offset += (n_in_tile % 32) * 16
                offset += (n_in_tile // 32) * _SCALE_GROUPS_PER_PANEL
                offset += group % _SCALE_GROUPS_PER_PANEL
                packed[offset] = source[row, group]
    return packed


def quantize_nvfp4_activations(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` activations to packed NVFP4 for native MMA."""
    if a.ndim != 2:
        raise ValueError(f"NVFP4 activations must be a matrix, got {tuple(a.shape)}")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"NVFP4 activations must be BF16/FP16, got {a.dtype}")
    m, k = a.shape
    if k % _GROUP_SIZE:
        raise ValueError(f"NVFP4 K={k} must be divisible by {_GROUP_SIZE}")
    groups = k // _GROUP_SIZE
    k_panels = triton.cdiv(groups, _SCALE_GROUPS_PER_PANEL)
    packed = torch.empty((m, k // 2), device=a.device, dtype=torch.uint8)
    scales = torch.ones(
        (nvfp4_scale_buffer_size(m, groups, operand="A"),),
        device=a.device,
        dtype=torch.float8_e4m3fn,
    )
    grid = (m, triton.cdiv(groups, _GROUPS_PER_PROGRAM))
    _pack_nvfp4_activations_kernel[grid](
        a,
        packed,
        scales,
        m,
        k,
        a.stride(0),
        a.stride(1),
        K_GROUPS=groups,
        K_PANELS=k_panels,
        GROUPS_PER_PROGRAM=_GROUPS_PER_PROGRAM,
        SCALE_GROUPS_PER_PANEL=_SCALE_GROUPS_PER_PANEL,
        SCALE_TILE_BYTES=_SCALE_TILE_BYTES,
        SCALE_A_TILE_ROWS=_SCALE_A_TILE_ROWS,
        num_warps=4,
    )
    return packed, scales


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


class _CuBLASLt:
    """ctypes binding for the small cuBLASLt API surface used here."""

    _CUDA_R_32F = 0
    _CUDA_R_16BF = 14
    _CUDA_R_4F_E2M1 = 33
    _CUBLAS_COMPUTE_32F = 68
    _MATMUL_DESC_POINTER_MODE = 2
    _MATMUL_DESC_A_SCALE_POINTER = 17
    _MATMUL_DESC_B_SCALE_POINTER = 18
    _MATMUL_DESC_A_SCALE_MODE = 31
    _MATMUL_DESC_B_SCALE_MODE = 32
    _MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 1
    _MATMUL_POINTER_MODE_DEVICE = 1
    _MATRIX_LAYOUT_ORDER = 1
    _ORDER_COL = 0
    _ORDER_ROW = 1
    _MATMUL_PREF_MAX_WORKSPACE_BYTES = 1

    def __init__(self) -> None:
        self.lib = self._load_library()
        self._bind_functions()
        self._handles: dict[int, ctypes.c_void_p] = {}
        self._zero: dict[int, torch.Tensor] = {}

    @staticmethod
    def _load_library() -> ctypes.CDLL:
        package_root = Path(torch.__file__).resolve().parent.parent
        cublas_dir = package_root / "nvidia" / "cublas" / "lib"
        runtime_dir = package_root / "nvidia" / "cuda_runtime" / "lib"
        library_path = cublas_dir / "libcublasLt.so.12"
        if not library_path.is_file():
            raise RuntimeError(
                "cuBLASLt NVFP4 backend needs nvidia-cublas-cu12 12.8 or newer"
            )

        # Wheel libraries do not always add their private dependency directory to
        # the process loader path. Preload their dependencies by absolute path.
        for dependency in (
            runtime_dir / "libcudart.so.12",
            cublas_dir / "libcublas.so.12",
        ):
            if dependency.is_file():
                try:
                    ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
        try:
            return ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"failed to load cuBLASLt from {library_path}: {exc}") from exc

    def _bind(self, name: str, argtypes: list[Any]) -> Any:
        function = getattr(self.lib, name)
        function.restype = ctypes.c_int
        function.argtypes = argtypes
        return function

    def _bind_functions(self) -> None:
        p = ctypes.c_void_p
        i = ctypes.c_int
        u64 = ctypes.c_uint64
        size = ctypes.c_size_t
        i64 = ctypes.c_int64
        self.create = self._bind("cublasLtCreate", [ctypes.POINTER(p)])
        self.matmul_desc_create = self._bind(
            "cublasLtMatmulDescCreate", [ctypes.POINTER(p), i, i]
        )
        self.matmul_desc_set = self._bind(
            "cublasLtMatmulDescSetAttribute", [p, i, p, size]
        )
        self.matmul_desc_destroy = self._bind("cublasLtMatmulDescDestroy", [p])
        self.layout_create = self._bind(
            "cublasLtMatrixLayoutCreate", [ctypes.POINTER(p), i, u64, u64, i64]
        )
        self.layout_set = self._bind("cublasLtMatrixLayoutSetAttribute", [p, i, p, size])
        self.layout_destroy = self._bind("cublasLtMatrixLayoutDestroy", [p])
        self.preference_create = self._bind(
            "cublasLtMatmulPreferenceCreate", [ctypes.POINTER(p)]
        )
        self.preference_set = self._bind(
            "cublasLtMatmulPreferenceSetAttribute", [p, i, p, size]
        )
        self.preference_destroy = self._bind("cublasLtMatmulPreferenceDestroy", [p])
        self.heuristic = self._bind(
            "cublasLtMatmulAlgoGetHeuristic",
            [p, p, p, p, p, p, p, i, ctypes.POINTER(_HeuristicResult),
             ctypes.POINTER(i)],
        )
        self.matmul = self._bind(
            "cublasLtMatmul", [p, p, p, p, p, p, p, p, p, p, p, p, p, p, size, p]
        )

    @staticmethod
    def _check(label: str, status: int) -> None:
        if status != 0:
            raise RuntimeError(f"cuBLASLt {label} failed with status {status}")

    def _set_int(self, function: Any, descriptor: ctypes.c_void_p,
                 attribute: int, value: int) -> None:
        scalar = ctypes.c_int(value)
        self._check(
            "set attribute",
            function(descriptor, attribute, ctypes.byref(scalar), ctypes.sizeof(scalar)),
        )

    def _set_pointer(self, descriptor: ctypes.c_void_p, attribute: int,
                     pointer: int) -> None:
        value = ctypes.c_void_p(pointer)
        self._check(
            "set pointer",
            self.matmul_desc_set(
                descriptor, attribute, ctypes.byref(value), ctypes.sizeof(value)
            ),
        )

    def _get_handle(self, device: int) -> ctypes.c_void_p:
        handle = self._handles.get(device)
        if handle is None:
            handle = ctypes.c_void_p()
            self._check("create handle", self.create(ctypes.byref(handle)))
            self._handles[device] = handle
        return handle

    def _get_zero(self, device: torch.device) -> torch.Tensor:
        index = int(device.index if device.index is not None else torch.cuda.current_device())
        zero = self._zero.get(index)
        if zero is None:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            self._zero[index] = zero
        return zero

    def _create_layout(self, dtype: int, rows: int, cols: int, leading_dim: int,
                       order: int = _ORDER_ROW) -> ctypes.c_void_p:
        descriptor = ctypes.c_void_p()
        self._check(
            "create matrix layout",
            self.layout_create(
                ctypes.byref(descriptor), dtype, rows, cols, leading_dim
            ),
        )
        self._set_int(self.layout_set, descriptor, self._MATRIX_LAYOUT_ORDER, order)
        return descriptor

    def create_plan(self, m: int, n: int, k: int, device: torch.device,
                    a_scale: torch.Tensor, b_scale: torch.Tensor) -> "_MatmulPlan":
        device_index = int(device.index if device.index is not None else torch.cuda.current_device())
        handle = self._get_handle(device_index)
        operation = ctypes.c_void_p()
        self._check(
            "create operation descriptor",
            self.matmul_desc_create(
                ctypes.byref(operation), self._CUBLAS_COMPUTE_32F, self._CUDA_R_32F
            ),
        )
        self._set_int(
            self.matmul_desc_set, operation, self._MATMUL_DESC_POINTER_MODE,
            self._MATMUL_POINTER_MODE_DEVICE,
        )
        self._set_int(
            self.matmul_desc_set, operation, self._MATMUL_DESC_A_SCALE_MODE,
            self._MATMUL_MATRIX_SCALE_VEC16_UE4M3,
        )
        self._set_int(
            self.matmul_desc_set, operation, self._MATMUL_DESC_B_SCALE_MODE,
            self._MATMUL_MATRIX_SCALE_VEC16_UE4M3,
        )
        self._set_pointer(operation, self._MATMUL_DESC_A_SCALE_POINTER, a_scale.data_ptr())
        self._set_pointer(operation, self._MATMUL_DESC_B_SCALE_POINTER, b_scale.data_ptr())

        a_layout = self._create_layout(self._CUDA_R_4F_E2M1, m, k, k)
        # The packed weight buffer is row-major [N, K]. Present the same bytes
        # as column-major [K, N] to cuBLASLt.
        b_layout = self._create_layout(
            self._CUDA_R_4F_E2M1, k, n, k, order=self._ORDER_COL
        )
        c_layout = self._create_layout(self._CUDA_R_16BF, m, n, n)
        d_layout = self._create_layout(self._CUDA_R_16BF, m, n, n)
        preference = ctypes.c_void_p()
        self._check(
            "create heuristic preference",
            self.preference_create(ctypes.byref(preference)),
        )
        max_workspace = ctypes.c_uint64(1 << 20)
        self._check(
            "set heuristic preference",
            self.preference_set(
                preference, self._MATMUL_PREF_MAX_WORKSPACE_BYTES,
                ctypes.byref(max_workspace), ctypes.sizeof(max_workspace),
            ),
        )
        results = (_HeuristicResult * 8)()
        returned = ctypes.c_int()
        status = self.heuristic(
            handle, operation, a_layout, b_layout, c_layout, d_layout,
            preference, len(results), results, ctypes.byref(returned),
        )
        self._check("algorithm query", status)
        candidates = [results[i] for i in range(returned.value) if results[i].state == 0]
        if not candidates:
            raise RuntimeError(
                f"cuBLASLt found no NVFP4 algorithm for M={m}, N={n}, K={k} "
                "within the 1 MiB workspace limit"
            )
        # cublasLt returns heuristic candidates in estimated-performance order;
        # the preference already caps every result at 1 MiB of workspace.
        chosen = candidates[0]
        c_input = torch.zeros((m, n), device=device, dtype=torch.bfloat16)
        plan = _MatmulPlan(
            library=self,
            handle=handle,
            operation=operation,
            a_layout=a_layout,
            b_layout=b_layout,
            c_layout=c_layout,
            d_layout=d_layout,
            preference=preference,
            # Copy the opaque algorithm descriptor out of the temporary
            # heuristic-result array before that array leaves this scope.
            algo=(ctypes.c_uint64 * 8)(*chosen.algo),
            workspace_size=chosen.workspace_size,
            device=device,
            c_input=c_input,
        )
        return plan


class _MatmulPlan:
    def __init__(self, library: _CuBLASLt, handle: ctypes.c_void_p,
                 operation: ctypes.c_void_p, a_layout: ctypes.c_void_p,
                 b_layout: ctypes.c_void_p, c_layout: ctypes.c_void_p,
                 d_layout: ctypes.c_void_p, preference: ctypes.c_void_p,
                 algo: ctypes.Array, workspace_size: int,
                 device: torch.device, c_input: torch.Tensor) -> None:
        self.library = library
        self.handle = handle
        self.operation = operation
        self.a_layout = a_layout
        self.b_layout = b_layout
        self.c_layout = c_layout
        self.d_layout = d_layout
        self.preference = preference
        self.algo = algo
        self.workspace_size = int(workspace_size)
        self.device = device
        self.workspace = (
            torch.empty((self.workspace_size,), device=device, dtype=torch.uint8)
            if self.workspace_size else None
        )
        self.c_input = c_input

    def run(self, a: torch.Tensor, a_scale: torch.Tensor, b: torch.Tensor,
            b_scale: torch.Tensor, b_global_scale: torch.Tensor,
            output: torch.Tensor, beta: torch.Tensor) -> None:
        library = self.library
        library._set_pointer(
            self.operation, library._MATMUL_DESC_A_SCALE_POINTER, a_scale.data_ptr()
        )
        library._set_pointer(
            self.operation, library._MATMUL_DESC_B_SCALE_POINTER, b_scale.data_ptr()
        )
        workspace_pointer = (
            ctypes.c_void_p(self.workspace.data_ptr()) if self.workspace is not None else None
        )
        stream = torch.cuda.current_stream(self.device).cuda_stream
        library._check(
            "matmul",
            library.matmul(
                self.handle,
                self.operation,
                ctypes.c_void_p(b_global_scale.data_ptr()),
                ctypes.c_void_p(a.data_ptr()),
                self.a_layout,
                ctypes.c_void_p(b.data_ptr()),
                self.b_layout,
                ctypes.c_void_p(beta.data_ptr()),
                ctypes.c_void_p(self.c_input.data_ptr()),
                self.c_layout,
                ctypes.c_void_p(output.data_ptr()),
                self.d_layout,
                ctypes.byref(self.algo),
                workspace_pointer,
                self.workspace_size,
                ctypes.c_void_p(stream),
            ),
        )

    def close(self) -> None:
        library = self.library
        library.preference_destroy(self.preference)
        library.matmul_desc_destroy(self.operation)
        for layout in (self.a_layout, self.b_layout, self.c_layout, self.d_layout):
            library.layout_destroy(layout)


_library: _CuBLASLt | None = None
_plan_cache: OrderedDict[tuple[int, int, int, int, int], _MatmulPlan] = OrderedDict()
_cache_lock = threading.RLock()


def cublaslt_nvfp4_gemm(
    a: torch.Tensor,
    b_nvfp4: torch.Tensor,
    b_scale: torch.Tensor,
    b_global_scale: torch.Tensor,
) -> torch.Tensor:
    """Run native NVFP4 MMA after packing activations to E2M1/E4M3 blocks."""
    if a.ndim != 2 or b_nvfp4.ndim != 2 or b_scale.ndim != 1:
        raise ValueError(
            "cuBLASLt NVFP4 GEMM expects 2D activation/weight and a 1D packed scale buffer"
        )
    if not a.is_cuda or not b_nvfp4.is_cuda:
        raise ValueError("cuBLASLt NVFP4 GEMM requires CUDA tensors")
    if any(tensor.device != a.device for tensor in (b_nvfp4, b_scale, b_global_scale)):
        raise ValueError("all cuBLASLt NVFP4 GEMM tensors must be on the same CUDA device")
    if torch.cuda.get_device_capability(a.device)[0] < 10:
        raise RuntimeError("native NVFP4 MMA requires a Blackwell-class GPU")
    if a.dtype != torch.bfloat16:
        raise TypeError(f"cuBLASLt NVFP4 GEMM expects BF16 activations, got {a.dtype}")
    if b_nvfp4.dtype != torch.uint8:
        raise TypeError(f"packed NVFP4 weights must use uint8 storage, got {b_nvfp4.dtype}")
    if b_scale.dtype != torch.float8_e4m3fn:
        raise TypeError(f"NVFP4 block scales must use E4M3 storage, got {b_scale.dtype}")
    if not b_nvfp4.is_contiguous() or not b_scale.is_contiguous():
        raise ValueError("packed NVFP4 weights and block scales must be contiguous")
    m, k = a.shape
    n = b_nvfp4.shape[0]
    if k % _GROUP_SIZE or b_nvfp4.shape != (n, k // 2):
        raise ValueError("packed NVFP4 weights must have shape [N, K//2] and K%16==0")
    expected_scale_groups = k // _GROUP_SIZE
    expected_b_scale_size = nvfp4_scale_buffer_size(
        n, expected_scale_groups, operand="B")
    if b_scale.numel() != expected_b_scale_size:
        raise ValueError(
            f"cuBLASLt weight scale buffer must contain {expected_b_scale_size} bytes"
        )
    if b_global_scale.numel() != 1 or b_global_scale.dtype != torch.float32:
        raise ValueError("NVFP4 global scale must be a scalar FP32 tensor")
    # MoE invokes each expert even when its routed token subset is empty. There
    # is no GEMM to schedule in that case, and cuBLASLt rejects a zero-M plan.
    if m == 0 or n == 0:
        return torch.empty((m, n), device=a.device, dtype=torch.bfloat16)

    a_packed, a_block_scale = quantize_nvfp4_activations(a)
    expected_a_scale_size = nvfp4_scale_buffer_size(
        m, expected_scale_groups, operand="A")
    if a_block_scale.numel() != expected_a_scale_size:
        raise RuntimeError("activation quantizer returned an invalid cuBLASLt scale layout")
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    device_index = int(a.device.index if a.device.index is not None else torch.cuda.current_device())
    key = (device_index, m, n, k, threading.get_ident())
    global _library
    with _cache_lock, torch.cuda.device(a.device):
        if _library is None:
            _library = _CuBLASLt()
        plan = _plan_cache.get(key)
        if plan is None:
            plan = _library.create_plan(m, n, k, a.device, a_block_scale, b_scale)
            _plan_cache[key] = plan
            if len(_plan_cache) > _MAX_CACHED_PLANS:
                _, oldest = _plan_cache.popitem(last=False)
                oldest.close()
        else:
            _plan_cache.move_to_end(key)
        beta = _library._get_zero(a.device)
        plan.run(a_packed, a_block_scale, b_nvfp4, b_scale,
                 b_global_scale, output, beta)
    return output
