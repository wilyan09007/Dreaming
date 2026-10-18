"""The veri.studio cell: the `finetune` candidate (S8 v2 `ft_encoder`) as a standalone script for a `veri run-script` job.

    veri run-script experiments/veri_finetune --entrypoint 'python train.py' --gpu-type L4-24GB --output-name dreaming-finetune-categorize

Unfreezes bge-small (BAAI/bge-small-en-v1.5) with a linear head on the `train` rows of rows.jsonl (rows.py writes it) and
saves, under $VERI_OUTPUT_DIR, the artifact predictors/finetune.py loads: `encoder/` in Hugging Face + sentence-transformers
layout (CLS pooling, normalize) and `head.npz` in predictors/student.py `Head.to_bytes` layout (W, b, labels, encoder_sha),
plus `artifact.tar`, the two packed as `finetune.pack` does (veri's download flattens subdirectories; the tar keeps them).
veri's image is Python 3.10 with torch, transformers and numpy: this file must not import the dreaming package.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("VERI_OUTPUT_DIR", HERE / "out"))
BASE, REV = "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # the hub commit student.py pins
BATCH, MAX_LEN, MIN_STEPS, MAX_STEPS, SCALE = 32, 128, 300, 600, 20.0  # as predictors/finetune.py: 8 epochs, floor 300, cap 600
ST_FILES = {  # the sentence-transformers wiring bge-small ships, so SentenceEncoder("encoder/") pools CLS and normalizes
    "modules.json": [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
                     {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                     {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}],
    "sentence_bert_config.json": {"max_seq_length": 512, "do_lower_case": True},
    "1_Pooling/config.json": {"word_embedding_dimension": 384, "pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False,
                              "pooling_mode_max_tokens": False, "pooling_mode_mean_sqrt_len_tokens": False},
}


def log(**kw: object) -> None:
    print(json.dumps(kw), flush=True)
    try:
        import veri_sdk  # in veri's image; absent on a laptop

        veri_sdk.log_metrics(**{k: v for k, v in kw.items() if isinstance(v, (int, float))})
    except Exception:  # noqa: BLE001  # metrics are a nicety, never the job
        pass


def hub_reachable() -> bool:
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    log(event="start", device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", torch=torch.__version__,
        hf_reachable=hub_reachable())
    rows = [json.loads(line) for line in (HERE / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = sorted({r["label"] for r in rows})
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] != "train"]
    src = str(HERE / "base") if (HERE / "base").is_dir() else BASE  # bundled copy if the code dir carries one
    kw = {} if src != BASE else {"revision": REV}
    tok = AutoTokenizer.from_pretrained(src, **kw)
    enc = AutoModel.from_pretrained(src, **kw).to(dev)
    head = torch.nn.Linear(enc.config.hidden_size, len(labels)).to(dev)

    def fwd(texts: list[str]) -> torch.Tensor:
        t = tok(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        return head(F.normalize(enc(**t).last_hidden_state[:, 0], dim=-1) * SCALE)

    per_epoch = math.ceil(len(train) / BATCH)
    steps = int(os.environ.get("STEPS", 0)) or min(MAX_STEPS, max(MIN_STEPS, 8 * per_epoch))
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 5e-5}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, (steps - s) / steps))
    y = torch.tensor([labels.index(r["label"]) for r in train], device=dev)
    t0, perm, losses = time.time(), [], []
    enc.train()
    for step in range(steps):
        if step % per_epoch == 0:
            perm = torch.randperm(len(train)).tolist()
        j = perm[(step % per_epoch) * BATCH :][:BATCH]
        loss = F.cross_entropy(fwd([train[i]["text"] for i in j]), y[j])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % per_epoch == 0 or step + 1 == steps:
            log(step=step + 1, loss=round(float(np.mean(losses)), 4))
            losses = []
    seconds = time.time() - t0
    enc.eval()
    with torch.no_grad():
        pred = [labels[int(k)] for i in range(0, len(held), 256) for k in fwd([r["text"] for r in held[i : i + 256]]).argmax(-1).cpu()]
    acc = float(np.mean([p == r["label"] for p, r in zip(pred, held)])) if held else 1.0
    log(event="done", train_rows=len(train), held_rows=len(held), steps=steps, train_seconds=round(seconds, 1), held_accuracy=round(acc, 4))

    (OUT / "encoder").mkdir(parents=True, exist_ok=True)
    enc.cpu().save_pretrained(OUT / "encoder")  # model.safetensors + config.json
    tok.save_pretrained(OUT / "encoder")
    for name, body in ST_FILES.items():
        (OUT / "encoder" / name).parent.mkdir(exist_ok=True)
        (OUT / "encoder" / name).write_text(json.dumps(body, indent=2))
    sha = hashlib.sha256((OUT / "encoder" / "model.safetensors").read_bytes()).hexdigest()
    W = (head.weight.detach().cpu().numpy().T * SCALE).astype(np.float32)  # head(normalize(h) * 20) == normalize(h) @ (20 W^T) + b
    np.savez(OUT / "head.npz", W=W, b=head.bias.detach().cpu().numpy().astype(np.float32), labels=np.array(labels), encoder_sha=np.array(sha))
    (OUT / "artifact.tar").write_bytes(pack(OUT / "encoder", (OUT / "head.npz").read_bytes()))
    log(event="saved", encoder_sha=sha, out=str(OUT))


def pack(encoder_dir: Path, head: bytes) -> bytes:
    """predictors/finetune.py `pack`, byte for byte. veri lists a job's output FLAT (basenames only, so encoder/config.json and
    1_Pooling/config.json collide and one is lost): the tar is the copy that survives `veri jobs download` with its layout."""
    files = [(f"encoder/{p.relative_to(encoder_dir).as_posix()}", p.read_bytes()) for p in encoder_dir.rglob("*") if p.is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name, blob in sorted([*files, ("head.npz", head)]):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(blob), 0o644
            tar.addfile(info, io.BytesIO(blob))
    return buf.getvalue()


if __name__ == "__main__":
    main()
