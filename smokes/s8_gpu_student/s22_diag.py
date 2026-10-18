import subprocess, json, ctypes, glob, os
out = {}
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60); return (r.stdout + r.stderr)[-1800:]
for name, c in dict(smi=r"nvidia-smi | head -20", dev="ls -la /dev/nvidia* /dev/dri 2>&1 | head -20", ver="cat /proc/driver/nvidia/version 2>&1 | head -3",
                    ld="ldconfig -p | grep -E 'libcuda|libnvidia-ml|libnvidia-ptx' | head", env="env | grep -i -E 'nvidia|cuda|ld_lib' ", cuda="ls -d /usr/local/cuda* 2>&1; ls /usr/local/cuda/compat 2>&1 | head -3",
                    libs="find / -name 'libcuda.so*' -not -path '*/proc/*' 2>/dev/null | head -20", pytorch="python -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' 2>&1 | tail -2",
                    nvmodprobe="which nvidia-modprobe nvidia-persistenced 2>&1; id").items(): out[name] = sh(c)
res = {}
for p in sorted(set(glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + glob.glob("/usr/local/nvidia/lib64/libcuda.so*") + glob.glob("/usr/local/cuda*/compat/libcuda.so*") + glob.glob("/usr/lib64/libcuda.so*"))):
    r = subprocess.run(["python", "-c", f"import ctypes;l=ctypes.CDLL('{p}');print(l.cuInit(0))"], capture_output=True, text=True); res[p] = (r.stdout + r.stderr)[-200:]
out["cuInit_by_lib"] = res
out["cuInit_default"] = sh("python -c \"import ctypes;l=ctypes.CDLL('libcuda.so.1');print(l.cuInit(0))\" 2>&1")
out["torch_env_variants"] = {v: sh(f"{v} python -c 'import torch;print(torch.cuda.is_available())' 2>&1 | tail -1") for v in ("CUDA_VISIBLE_DEVICES=0", "CUDA_VISIBLE_DEVICES=GPU-0", "NVIDIA_VISIBLE_DEVICES=all CUDA_VISIBLE_DEVICES=0")}
print(json.dumps(out, indent=1)); json.dump(out, open("results.json", "w"), indent=1)
