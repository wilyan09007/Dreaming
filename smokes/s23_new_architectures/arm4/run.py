"""S23 arm 4: attention-head reweighting (one learned scalar per attention head, init 1, own implementation of the AHR idea via forward pre-hooks on o_proj)
vs LoRA r=16 (all linear layers) on Qwen2.5-0.5B, same step budget, k in {3, 10, 30}. Two variants:
  cls = AutoModelForSequenceClassification (new linear head trained in every method; 'head only' control included)
  lm  = paper-style prompt that lists the labels as letters, class score = LM-head logit of the letter token (no new parameters)
Learning rates are picked per method on a DEV task (CLINC 'home' domain, k=10, draw 0), never on the reported tasks. Usage: python run.py [--tiny] [--deadline-min N]"""
import sys, argparse, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import *
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

OUT = Path(__file__).resolve().parent / "rows.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B"; DEV = "cuda" if torch.cuda.is_available() else "cpu"; LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STEPS, BS = 100, 16
TOK = AutoTokenizer.from_pretrained(MODEL); TOK.padding_side = "left"
LETTER_IDS = [TOK.encode(" " + c, add_special_tokens=False)[0] for c in LET[:15]]
assert all(len(TOK.encode(" " + c, add_special_tokens=False)) == 1 for c in LET[:15])

def texts_for(t, variant, rows):
    if variant == "cls": return list(rows)
    head = "Classify the utterance into exactly one intent.\n" + "\n".join(f"{LET[i]}. {n}" for i, n in enumerate(t["labels"])) + "\nUtterance: "
    return [head + r + "\nAnswer:" for r in rows]

def load_base(variant, K):
    if variant == "cls": m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=K, torch_dtype=torch.float32)
    else: m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.config.pad_token_id = TOK.pad_token_id; return m.to(DEV)

def add_ahr(m):
    """one scalar per attention head, scale = 1 + delta, delta init 0; multiplies each head's output right before o_proj"""
    cfg = m.config; H, d, L = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.num_hidden_layers
    delta = nn.Parameter(torch.zeros(L, H, device=DEV))
    for i, layer in enumerate(m.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(lambda mod, args, i=i: ((args[0].unflatten(-1, (H, d)) * (1 + delta[i]).view(1, 1, H, 1).to(args[0].dtype)).flatten(-2),))
    return delta

def logits_of(m, variant, enc, K):
    if variant == "cls": return m(**enc).logits.float()
    return m(**enc, logits_to_keep=1).logits[:, -1][:, LETTER_IDS[:K]].float()

def encode_batch(txts, variant):
    return {k_: v.to(DEV) for k_, v in TOK(txts, return_tensors="pt", padding=True, truncation=True, max_length=96 if variant == "cls" else 320).items()}

@torch.no_grad()
def predict(m, variant, txts, K, bs=64):
    m.eval(); out = []
    for i in range(0, len(txts), bs):
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"): out.append(torch.softmax(logits_of(m, variant, encode_batch(txts[i:i + bs], variant), K), -1).cpu().numpy())
    return np.concatenate(out)

def run(base, variant, method, t, Ltxt, Ly, Qtxt, lr, head_lr, steps, seed):
    K = len(t["labels"]); torch.manual_seed(seed); m = copy.deepcopy(base)
    if variant == "cls":
        g = torch.Generator().manual_seed(seed); m.score.weight.data = (torch.randn(m.score.weight.shape, generator=g) * 0.02).to(DEV)
    for p in m.parameters(): p.requires_grad = False
    delta = None
    if method == "head": m.score.weight.requires_grad = True
    elif method == "ahr":
        delta = add_ahr(m)
        if variant == "cls": m.score.weight.requires_grad = True
    elif method == "lora":
        from peft import LoraConfig, get_peft_model
        m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                                         task_type="SEQ_CLS" if variant == "cls" else "CAUSAL_LM"))
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad] + ([("delta", delta)] if delta is not None else [])
    heads = [p for n, p in named if "score" in n]; rest = [p for n, p in named if "score" not in n]
    opt = torch.optim.AdamW(([{"params": heads, "lr": head_lr}] if heads else []) + ([{"params": rest, "lr": lr}] if rest else []), weight_decay=0.0)
    ntrain = sum(p.numel() for _, p in named); y = torch.tensor(Ly, device=DEV); rng = np.random.default_rng(seed); t0 = time.time(); m.train()
    for _ in range(steps):
        idx = rng.choice(len(Ltxt), min(BS, len(Ltxt)), replace=False)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"): lg = logits_of(m, variant, encode_batch([Ltxt[i] for i in idx], variant), K)
        loss = F.cross_entropy(lg, y[idx]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0); opt.step()
    secs = time.time() - t0; P = predict(m, variant, Qtxt, K); del m, opt
    if DEV == "cuda": torch.cuda.empty_cache()
    return P, ntrain, secs

def dev_select(variant, tiny):
    """pick per-method lrs on the DEV task (CLINC 'home', k=10, draw 0)"""
    if tiny: return dict(head_lr=2e-3, ahr=1e-2, lora=3e-4), {}
    d = clinc_task("home"); K = len(d["labels"]); L = draw(d["ys"], 10, 0, K); base = load_base(variant, K)
    Ltxt, Ly = texts_for(d, variant, [d["ts"][i] for i in L]), d["ys"][L]; Qtxt = texts_for(d, variant, d["tq"]); acc = lambda P: float((P.argmax(1) == d["yq"]).mean())
    sel = {"head_lr": 2e-3}; res = {}
    if variant == "cls":
        for hl in (5e-4, 2e-3, 1e-2): res[f"head@{hl}"] = acc(run(base, variant, "head", d, Ltxt, Ly, Qtxt, 0, hl, STEPS, 0)[0]); log("dev", f"head@{hl}", res[f"head@{hl}"])
        sel["head_lr"] = float(max((h for h in (5e-4, 2e-3, 1e-2)), key=lambda h: res[f"head@{h}"]))
    for meth, grid in (("ahr", (3e-3, 1e-2, 3e-2, 1e-1)), ("lora", (1e-4, 3e-4, 1e-3))):
        for lr in grid: res[f"{meth}@{lr}"] = acc(run(base, variant, meth, d, Ltxt, Ly, Qtxt, lr, sel["head_lr"], STEPS, 0)[0]); log("dev", f"{meth}@{lr}", res[f"{meth}@{lr}"])
        sel[meth] = float(max(grid, key=lambda l: res[f"{meth}@{l}"]))
    del base; return sel, res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); ap.add_argument("--deadline-min", type=float, default=85); a = ap.parse_args()
    if OUT.exists(): OUT.unlink()
    T = tasks(tiny=a.tiny); ks, draws, steps = ([3], 1, 2) if a.tiny else ([3, 10, 30], DRAWS, STEPS); deadline = time.time() + a.deadline_min * 60
    if a.tiny:
        for t in T: t["tq"], t["yq"], t["Xq"] = t["tq"][:30], t["yq"][:30], t["Xq"][:30]
    for variant in ("cls", "lm"):
        sel, dres = dev_select(variant, a.tiny); log(variant, "selected", sel); emit(OUT, arm="4", variant=variant, kind="dev_selection", sel=sel, dev=dres)
        bases = {}
        for dr in range(draws):
            for t in T:
                K = len(t["labels"])
                if K not in bases: bases[K] = load_base(variant, K)
                base = bases[K]; Qtxt = texts_for(t, variant, t["tq"])
                if variant == "lm" and dr == 0:
                    P = predict(base, variant, Qtxt, K); emit(OUT, arm="4", variant=variant, method="zero-shot (no training)", task=t["name"], family=t["family"], K=K, k=0, draw=0, params=0, **score(P, t["yq"]))
                for k in ks:
                    if time.time() > deadline: log("deadline reached"); break
                    L = draw(t["ys"], k, dr, K); Ltxt, Ly = texts_for(t, variant, [t["ts"][i] for i in L]), t["ys"][L]
                    if variant == "cls":
                        clf = lr_fit(t["Xs"][L], Ly); emit(OUT, arm="4", variant="bge", method="LR baseline (frozen bge-small)", task=t["name"], family=t["family"], K=K, k=k, draw=dr, params=0, **score(full_proba(clf, t["Xq"], K), t["yq"]))
                    for meth in (("head", "ahr", "lora") if variant == "cls" else ("ahr", "lora")):
                        P, n, secs = run(base, variant, meth, t, Ltxt, Ly, Qtxt, sel.get(meth, 0), sel["head_lr"], steps, dr)
                        s = score(P, t["yq"]); emit(OUT, arm="4", variant=variant, method={"head": "head only", "ahr": "AHR (scalar per head)", "lora": "LoRA r=16"}[meth], task=t["name"], family=t["family"], K=K, k=k, draw=dr,
                                                    params=n, train_secs=secs, lr=sel.get(meth), steps=steps, **s)
                        log(variant, t["name"], f"k{k} d{dr}", meth, f"acc {s['acc']:.3f} cov {s['cov']:.2f} params {n} {secs:.1f}s")
        del bases
    log("done", OUT)
