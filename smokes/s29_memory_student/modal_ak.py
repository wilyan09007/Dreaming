"""S29 fallback for when RunPod has no GPU supply: run sections on Modal CPU instead of a pod (used for A_k, exact k rows per label).
    cd mock-dir && uv run modal run ../smokes/s29_memory_student/modal_ak.py --sections "A_k"      (MODAL tokens in the environment)
Ships run.py, common.py and the two caches, runs `run.py <sections>` with S29_CPU=1 (skip the CUDA gate), writes results.json and log.txt to rerun/.
Image pins mirror the trainer's (apps/trainer/src/dreaming_trainer/modal_app.py), plus what run.py imports."""
import pathlib
import modal

HERE = pathlib.Path(__file__).resolve().parent
SMOKES = HERE.parent
image = (
    modal.Image.debian_slim(python_version="3.12.10")
    .uv_pip_install("torch==2.14.1", "sentence-transformers==6.1.0", "numpy==2.5.4", "scipy==1.17.1", "psycopg[binary]==3.3.6", "pydantic==2.14.1")
    .uv_pip_install("scikit-learn==1.9.1", "datasets==5.1.0")  # no hnswlib: E's index tests report "not installed" here
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("dreaming-s29", image=image)


@app.function(cpu=8.0, memory=16384, timeout=40 * 60)
def run(files: dict[str, bytes], sections: list[str]) -> dict[str, bytes]:
    import os, subprocess, sys
    root = pathlib.Path("/proj")
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True); (root / rel).write_bytes(data)
    out = pathlib.Path("/tmp/out"); out.mkdir(exist_ok=True)
    p = subprocess.run([sys.executable, "smokes/s29_memory_student/run.py", *sections], cwd=root, env=dict(os.environ, S29_OUT=str(out), S29_CPU="1"))
    return {f.name: f.read_bytes() for f in out.iterdir() if f.is_file()} | {"exit.txt": str(p.returncode).encode()}


@app.local_entrypoint()
def main(sections: str = "A_k"):
    rels = ["common.py", "s29_memory_student/run.py", ".cache/label_descriptions.json", ".cache/synth_examples.json"]
    got = run.remote({"smokes/" + r: (SMOKES / r).read_bytes() for r in rels}, sections.split())
    dest = HERE / "rerun"; dest.mkdir(exist_ok=True)
    for name, data in got.items(): (dest / name).write_bytes(data)
    print("wrote", sorted(got), "to", dest, "exit", got["exit.txt"].decode())
