"""阶段2侦察：DeepSeek 模型支持/本地权重/网络可得性一次性摸清。"""
import os, sys, socket, time

print("== python / torch / transformers ==")
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("python", sys.version.split()[0])

print("\n== transformers deepseek 模块 ==")
mods = [m for m in ("modeling_deepseek", "modeling_deepseek_v2", "modeling_deepseek_v3")]
tf_dir = os.path.dirname(transformers.__file__)
print("transformers dir:", tf_dir)
for m in mods:
    p = os.path.join(tf_dir, "models", m + ".py")
    print(f"{m}: {'EXISTS' if os.path.exists(p) else 'missing'} ({p})")
import transformers.models as M
print("models dir has deepseek*:", [d for d in os.listdir(os.path.join(tf_dir, "models")) if "deep" in d.lower()])
from transformers import AutoConfig
print("AutoConfig keys check...")
for k in ("DeepseekV2Config", "DeepseekConfig", "DeepseekV3Config"):
    print(k, hasattr(M, k) or k in dir(M))

print("\n== 本地模型目录 ==")
home = os.path.expanduser("~")
for root in ("huggingface",):
    d = os.path.join(home, root)
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            full = os.path.join(d, name)
            print(f"{root}/{name}  {'(dir)' if os.path.isdir(full) else ''}")
            if os.path.isdir(full):
                subs = sorted(os.listdir(full))[:6]
                print("   ", subs)

print("\n== GPU ==")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
free, _ = torch.cuda.mem_get_info()
print(f"free {free/1e9:.2f} GB")

print("\n== 网络（huggingface.co）==")
def net(host, port=443, t=6):
    try:
        s = socket.create_connection((host, port), timeout=t)
        s.close()
        return True
    except Exception as e:
        return f"{type(e).__name__}"
for h in ("huggingface.co", "cdn-lfs.huggingface.co"):
    print(h, "->", net(h))
