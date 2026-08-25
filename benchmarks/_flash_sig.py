"""打印 flash-attn 的 window_size / block_table 相关签名与文档（SWA 实现前钉死约定）。

用法: python benchmarks/_flash_sig.py
"""
import inspect

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


def main():
    for fn in (flash_attn_varlen_func, flash_attn_with_kvcache):
        print(f"==== {fn.__module__}.{fn.__name__} ====")
        print("signature:", inspect.signature(fn))
        doc = fn.__doc__ or ""
        # 打印与 window/block_table/position 相关的文档段落
        for line in doc.splitlines():
            low = line.lower()
            if any(k in low for k in ("window", "block_table", "cache_seqlens",
                                      "left", "right", "position")):
                print("  |", line.strip())
        print()


if __name__ == "__main__":
    main()
