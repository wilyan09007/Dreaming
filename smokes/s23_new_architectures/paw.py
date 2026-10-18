"""S23-lite arm 3 (GPU): Programs-as-Weights (PAW) public compiler as a spec -> LoRA hypernetwork.
Pipeline = the official compile.py (MIT, vendored unmodified as paw_compile.py; its own functions do every step, we only batch the generation):
 spec (label list + descriptions + k examples) -> pseudo-program (Qwen3-4B-Instruct) -> compiler states (programasweights/paw-4b-qwen3-0.6b, 4B) -> rank-64 LoRA for the frozen Qwen3-0.6B interpreter.
Classes are lettered (A, B, ...) in the spec so one forward pass gives a probability per class (softmax over the letter-token logits at the answer position).
Rows: (a) zero-shot accuracy of the compiled adapter (k=0 spec, and spec with k examples), no-adapter control, (b) fine-tune from the compiled adapter vs from the default
random LoRA init (B=0) with the same prompt, rows, steps and lr. Cells run draw-major so a timeout leaves whole draws.
Usage: python paw.py [--selftest] [--deadline-min N]"""
import sys, argparse, json, time, re, traceback
from lib import *
import torch, torch.nn.functional as F
import paw_compile as pc

OUT = HERE / "rows_paw.jsonl"; PSEUDO = HERE / "pseudo_programs.jsonl"; STATUS = HERE / "paw_status.json"
CFG = pc.COMPILERS["standard"]; LET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, BS, LR, SPEC_TOK_CAP, EVAL_BS = 60, 8, 5e-4, 3600, 48
status = {}
def set_status(k, v):
    status[k] = v; STATUS.write_text(json.dumps(status, indent=1), encoding="utf-8"); log("STATUS", k, str(v)[:300])

def fp(cls, name, **kw):  # transformers renamed torch_dtype -> dtype
    try: return cls.from_pretrained(name, dtype=torch.bfloat16, **kw)
    except TypeError: return cls.from_pretrained(name, torch_dtype=torch.bfloat16, **kw)

# ------------------------------------------------------------------ specs
def spec_text(t, ex, mode):
    K = len(t["labels"])
    if mode == "letters":
        lines = [f"Classify one short user message into exactly one of {K} intents. Reply with the single capital letter of the intent only, nothing else.", "", "Intents:"]
        lines += [f"{LET[i]}. {n}: {d}" for i, (n, d) in enumerate(zip(t["labels"], t["desc"]))]
    else:
        lines = [f"Classify one short user message into exactly one of {K} intents. Reply with the intent name only, written exactly as below.", "", "Intents:"]
        lines += [f"{n}: {d}" for n, d in zip(t["labels"], t["desc"])]
    if ex:
        lines += ["", "Labelled examples:"] + [f"Message: {x[:200]}\nAnswer: {LET[y] if mode == 'letters' else t['labels'][y]}" for x, y in ex]
    return "\n".join(lines)

def build_specs(T, ctok, draws, ks):
    specs, rng = {}, np.random.default_rng(7)
    def add(key, t, L, mode, n_per):
        while True:
            ex = [(t["ts"][i], c) for c in range(len(t["labels"])) for i in L[t["ys"][L] == c][:n_per]] if len(L) else []
            txt = spec_text(t, [ex[j] for j in rng.permutation(len(ex))], mode); n = len(ctok.encode(txt))
            if n <= SPEC_TOK_CAP or n_per <= 1: break
            n_per -= 1
        specs[key] = dict(text=txt, task=t["name"], mode=mode, n_per=n_per if len(L) else 0, tokens=n)
    for t in T:
        K = len(t["labels"]); empty = np.array([], int); add(f"{t['name']}|names|0|0", t, empty, "names", 0); add(f"{t['name']}|letters|0|0", t, empty, "letters", 0)
        for dr in range(draws):
            for k in ks: add(f"{t['name']}|letters|{k}|{dr}", t, draw(t["ys"], k, dr, K), "letters", min(k, 10))
    return specs

# ------------------------------------------------------------------ compile stages (paw_compile's own functions)
@torch.inference_mode()
def gen_pseudo(specs, dev):
    cache = {}
    if PSEUDO.exists():
        for ln in PSEUDO.read_text(encoding="utf-8").splitlines(): r = json.loads(ln); cache[r["key"]] = r["pseudo"]
    todo = [k for k in specs if k not in cache]
    if todo:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(pc.REFERENCE_MODEL, revision=pc.REFERENCE_REVISION); tok.padding_side = "left"
        model = fp(AutoModelForCausalLM, pc.REFERENCE_MODEL, revision=pc.REFERENCE_REVISION, attn_implementation="sdpa").to(dev).eval()
        prompts = {k: pc.render_chat(tok, pc.compiler_prompt_examples(specs[k]["text"])) for k in todo}
        lens = {k: len(tok.encode(prompts[k], add_special_tokens=False)) for k in todo}; todo.sort(key=lambda k: lens[k]); t0 = time.time()
        def run(keys):
            enc = tok([prompts[k] for k in keys], add_special_tokens=False, return_tensors="pt", padding=True).to(dev); n_in = enc["input_ids"].shape[1]
            out = model.generate(**enc, do_sample=False, max_new_tokens=pc.pseudo_token_budget(max(lens[k] for k in keys)), pad_token_id=tok.eos_token_id)
            for j, k in enumerate(keys): cache[k] = tok.decode(out[j, n_in:], skip_special_tokens=True).strip(); emit(PSEUDO, key=k, pseudo=cache[k])
        i = 0
        while i < len(todo):
            bs = max(1, min(8, 16000 // lens[todo[min(i + 7, len(todo) - 1)]])); keys = todo[i:i + bs]; i += bs
            try: run(keys)
            except Exception as e:  # OOM or similar: one by one
                log("batch generation failed, retrying one by one:", repr(e)[:150]); pc.release_gpu()
                for k in keys:
                    try: run([k])
                    except Exception as e2: cache[k] = ""; log("generation failed for", k, repr(e2)[:150])
            log("pseudo-programs", i, "/", len(todo), f"{time.time()-t0:.0f}s")
        del model; pc.release_gpu()
    return {k: cache[k] for k in specs}

@torch.inference_mode()
def encode_all(specs, pseudo, ckpt, meta, dev):
    from transformers import AutoModel, AutoTokenizer
    ctok = AutoTokenizer.from_pretrained(ckpt / "compiler"); cm = fp(AutoModel, ckpt / "compiler", attn_implementation="sdpa").to(dev).eval()
    pooled, bad = {}, {}
    for k, s in specs.items():
        try:
            ids = pc.build_compiler_input(ctok, s["text"], pseudo[k], meta["prefix_steps"])
            o = cm(input_ids=torch.tensor([ids], device=dev), output_hidden_states=True, use_cache=False)
            pooled[k] = pc.pool_prefix(o.hidden_states, CFG["layers"], meta["prefix_steps"]).cpu(); del o
        except Exception as e: bad[k] = repr(e)[:200]
    del cm; pc.release_gpu(); return pooled, bad

# ------------------------------------------------------------------ interpreter side
def inject(pm, params):
    root = pm.base_model.model
    for (layer, name), (a, b) in params.items():
        sub = root.get_submodule(CFG["modules"][name][1].format(layer)); A, B = sub.lora_A["default"].weight, sub.lora_B["default"].weight
        assert A.shape == a.shape and B.shape == b.shape, (A.shape, a.shape, B.shape, b.shape)
        A.data.copy_(a); B.data.copy_(b)

def rand_init(pm, seed):
    torch.manual_seed(seed)
    for m in pm.modules():
        if hasattr(m, "lora_A") and hasattr(m, "reset_lora_parameters"): m.reset_lora_parameters("default", True)

def fill(template, text): return template.replace(pc.INPUT_PLACEHOLDER, text.replace(pc.INPUT_PLACEHOLDER, ""))

def enc_batch(tok, prompts): return {k: v.to(DEV) for k, v in tok(prompts, add_special_tokens=False, return_tensors="pt", padding=True).items()}

@torch.no_grad()
def eval_letters(pm, tok, template, texts, K, ids):
    pm.eval(); out = []
    for i in range(0, len(texts), EVAL_BS):
        lg = pm(**enc_batch(tok, [fill(template, x) for x in texts[i:i + EVAL_BS]]), logits_to_keep=1).logits[:, -1][:, ids[:K]].float()
        out.append(torch.softmax(lg, -1).cpu().numpy())
    return np.concatenate(out)

def finetune(pm, tok, template, texts, y, K, ids, steps, lr, seed):
    params = [p for p in pm.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed); yt = torch.tensor(y, device=DEV); pm.train(); t0 = time.time()
    for _ in range(steps):
        idx = rng.choice(len(texts), min(BS, len(texts)), replace=False)
        lg = pm(**enc_batch(tok, [fill(template, texts[i]) for i in idx]), logits_to_keep=1).logits[:, -1][:, ids[:K]].float()
        loss = F.cross_entropy(lg, yt[idx]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    pm.eval()
    if DEV == "cuda": torch.cuda.synchronize()
    return time.time() - t0

@torch.no_grad()
def gen_names(pm, tok, template, texts, labels, bs=32):
    norm = lambda s: " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("_", " ")).split()); lab = [norm(x) for x in labels]; pred = []; pm.eval()
    for i in range(0, len(texts), bs):
        enc = enc_batch(tok, [fill(template, x) for x in texts[i:i + bs]]); n_in = enc["input_ids"].shape[1]
        g = pm.generate(**enc, do_sample=False, max_new_tokens=16, pad_token_id=tok.pad_token_id)
        for o in g: s = norm(tok.decode(o[n_in:], skip_special_tokens=True)); pred.append(lab.index(s) if s in lab else -1)
    return np.array(pred)

def load_interpreter(rank, alpha, selftest=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    if selftest:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        base = Qwen3ForCausalLM(Qwen3Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=len(tok) + 64)).to(torch.bfloat16)
    else:
        tok = AutoTokenizer.from_pretrained(CFG["interpreter"], revision=pc.INTERPRETER_REVISION)
        base = fp(AutoModelForCausalLM, CFG["interpreter"], revision=pc.INTERPRETER_REVISION)
    tok.padding_side = "left"
    pm = get_peft_model(base.to(DEV), LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=sorted(CFG["modules"]), bias="none", task_type="CAUSAL_LM"))
    return pm, tok

def selftest():
    """CPU check of the interpreter-side code with a random 2-layer Qwen3 and fake compiled weights (no 4B model, no big downloads)."""
    T = load_tasks(); t = T[0]; K = len(t["labels"]); pm, tok = load_interpreter(8, 16, selftest=True)
    ids = [tok.encode(c, add_special_tokens=False)[0] for c in LET]; tmpl = "Q: " + pc.INPUT_PLACEHOLDER + "\nA:"; params = {}
    for L in range(2):
        for name in CFG["modules"]:
            sub = pm.base_model.model.get_submodule(CFG["modules"][name][1].format(L)); params[L, name] = (torch.randn_like(sub.lora_A["default"].weight.float()).bfloat16(), torch.randn_like(sub.lora_B["default"].weight.float()).bfloat16())
    inject(pm, params); P1 = eval_letters(pm, tok, tmpl, t["tq"][:20], K, ids); rand_init(pm, 0); P0 = eval_letters(pm, tok, tmpl, t["tq"][:20], K, ids)
    assert P1.shape == (20, K) and abs(P1.sum(1) - 1).max() < 1e-4 and not np.allclose(P0, P1), "inject / rand_init"
    L = draw(t["ys"], 3, 0, K); secs = finetune(pm, tok, tmpl, [t["ts"][i] for i in L], t["ys"][L], K, ids, 3, 1e-3, 0); P2 = eval_letters(pm, tok, tmpl, t["tq"][:20], K, ids)
    assert not np.allclose(P0, P2); g = gen_names(pm, tok, tmpl, t["tq"][:4], t["labels"]); assert g.shape == (4,)
    sp = build_specs(T, tok, 1, [1, 3]); print({k: (v["tokens"], v["n_per"]) for k, v in sp.items()}); print(list(sp.values())[3]["text"][:500])
    s = score(P2, t["yq"][:20], P2, t["yq"][:20]); print("selftest ok", f"{secs:.1f}s", s)

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--deadline-min", type=float, default=30); a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    if not torch.cuda.is_available(): print("NO CUDA"); sys.exit(2)
    t_start = time.time(); deadline = t_start + a.deadline_min * 60
    if OUT.exists(): OUT.unlink()
    try:
        from huggingface_hub import snapshot_download
        ckpt = Path(snapshot_download(CFG["checkpoint"], revision=CFG["revision"], allow_patterns=["meta.json", "lora_mapper.pt", "compiler/*"])); meta = json.loads((ckpt / "meta.json").read_text())
        set_status("checkpoint", f"{CFG['checkpoint']}@{CFG['revision']} rank={meta['lora_rank']} alpha={meta['lora_alpha']} prefix={meta['prefix_steps']} ready at t={time.time()-t_start:.0f}s")
        from transformers import AutoTokenizer
        T = load_tasks(); ctok = AutoTokenizer.from_pretrained(ckpt / "compiler"); specs = build_specs(T, ctok, DRAWS, KS)
        set_status("specs", f"{len(specs)} specs, max tokens {max(s['tokens'] for s in specs.values())}, examples/label in k=10 specs: {sorted({s['n_per'] for k, s in specs.items() if '|letters|10|' in k})}")
        t0 = time.time(); pseudo = gen_pseudo(specs, DEV)
        set_status("pseudo", f"{len(pseudo)} pseudo-programs in {time.time()-t0:.0f}s; empty={sum(1 for v in pseudo.values() if not v)}; with closing marker={sum('[END_PSEUDO_PROGRAM]' in v for v in pseudo.values())}")
        t0 = time.time(); pooled, bad = encode_all(specs, pseudo, ckpt, meta, DEV); set_status("compiler", f"{len(pooled)} encoded in {time.time()-t0:.0f}s, {len(bad)} failed {list(bad.items())[:2]}")
        mapper = pc.LoraMapper(torch.load(ckpt / "lora_mapper.pt", map_location="cpu", weights_only=True))
        pm, tok = load_interpreter(meta["lora_rank"], meta["lora_alpha"]); ids = [tok.encode(c, add_special_tokens=False)[0] for c in LET]
        assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LET[:15])
        templates = {}
        for k in specs:
            try: templates[k] = pc.render_interpreter_prompt(pseudo[k], CFG)
            except Exception as e: bad[k] = repr(e)[:200]
        NP = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        set_status("interpreter", f"loaded, trainable LoRA params {NP}, templates {len(templates)}, bad {len(bad)}, ready at t={time.time()-t_start:.0f}s")
    except Exception:
        set_status("FAILED_SETUP", traceback.format_exc()[-1500:]); sys.exit(1)
    byname = {t["name"]: t for t in T}
    def row(method, t, k, dr, key, Pq, Pc=None, **kw): emit(OUT, arm="3", method=method, task=t["name"], family=t["family"], K=len(t["labels"]), k=k, draw=dr, spec_n_per=specs[key]["n_per"], **kw, **score(Pq, t["yq"], Pc, t["yc"]))
    def eval_cal_test(t, tmpl, K):
        P = eval_letters(pm, tok, tmpl, t["tc"] + t["tq"], K, ids); return P[len(t["tc"]):], P[:len(t["tc"])]
    cell_secs = []
    try:
        # (a) zero-shot at k=0: letters (probabilities) and names (greedy text)
        for t in T:
            K = len(t["labels"]); kl, kn = f"{t['name']}|letters|0|0", f"{t['name']}|names|0|0"
            if kl in templates and kl in pooled:
                t0 = time.time(); inject(pm, mapper(pooled[kl])); row("PAW compiled adapter, zero-shot (k=0 spec)", t, 0, 0, kl, eval_letters(pm, tok, templates[kl], t["tq"], K, ids), params=NP, secs=time.time() - t0)
                rand_init(pm, 0); row("no adapter, same prompt (k=0 spec)", t, 0, 0, kl, eval_letters(pm, tok, templates[kl], t["tq"], K, ids), params=0)
            if kn in templates and kn in pooled:
                inject(pm, mapper(pooled[kn])); pr = gen_names(pm, tok, templates[kn], t["tq"], t["labels"])
                emit(OUT, arm="3", method="PAW compiled adapter, zero-shot (k=0 spec, greedy label text)", task=t["name"], family=t["family"], K=K, k=0, draw=0, acc=float((pr == t["yq"]).mean()), valid=float((pr >= 0).mean()), params=NP)
            log("zero-shot", t["name"], f"t={time.time()-t_start:.0f}s")
        # (b) k > 0, draw-major: zero-shot of the compiled adapter (spec with k examples), then fine-tune compiled init vs random init, same prompt / steps / lr / rows
        for dr in range(DRAWS):
            for t in T:
                K = len(t["labels"])
                for k in KS:
                    if time.time() + 1.3 * (max(cell_secs) if cell_secs else 90) > deadline: log("deadline reached"); raise StopIteration
                    c0 = time.time(); key = f"{t['name']}|letters|{k}|{dr}"
                    if key not in templates or key not in pooled: continue
                    L = draw(t["ys"], k, dr, K); Ltxt, Ly = [t["ts"][i] for i in L], t["ys"][L]; tmpl = templates[key]; params = mapper(pooled[key])
                    inject(pm, params); row("PAW compiled adapter, zero-shot (spec with k examples)", t, k, dr, key, eval_letters(pm, tok, tmpl, t["tq"], K, ids), params=NP)
                    secs = finetune(pm, tok, tmpl, Ltxt, Ly, K, ids, STEPS, LR, dr); Pq, Pc = eval_cal_test(t, tmpl, K)
                    row("PAW compiled init + fine-tune", t, k, dr, key, Pq, Pc, params=NP, train_secs=secs, steps=STEPS, lr=LR)
                    rand_init(pm, dr); secs = finetune(pm, tok, tmpl, Ltxt, Ly, K, ids, STEPS, LR, dr); Pq, Pc = eval_cal_test(t, tmpl, K)
                    row("random LoRA init + fine-tune (same prompt)", t, k, dr, key, Pq, Pc, params=NP, train_secs=secs, steps=STEPS, lr=LR)
                    if dr == 0:
                        rand_init(pm, 0); row("no adapter, same prompt (spec with k examples)", t, k, dr, key, eval_letters(pm, tok, tmpl, t["tq"], K, ids), params=0)
                    cell_secs.append(time.time() - c0); log("cell", key, f"{cell_secs[-1]:.0f}s t={time.time()-t_start:.0f}s")
        # (c) lr sensitivity of the compiled-vs-random comparison on two tasks (k=10, draw 0)
        for t in [T[0], T[min(2, len(T) - 1)]]:
            K = len(t["labels"]); key = f"{t['name']}|letters|10|0"
            if key not in templates: continue
            L = draw(t["ys"], 10, 0, K); Ltxt, Ly = [t["ts"][i] for i in L], t["ys"][L]
            for lr in (1e-4, 2e-3):
                if time.time() + 1.3 * (max(cell_secs) if cell_secs else 90) > deadline: raise StopIteration
                inject(pm, mapper(pooled[key])); secs = finetune(pm, tok, templates[key], Ltxt, Ly, K, ids, STEPS, lr, 0); Pq, Pc = eval_cal_test(t, templates[key], K)
                row("PAW compiled init + fine-tune", t, 10, 0, key, Pq, Pc, params=NP, train_secs=secs, steps=STEPS, lr=lr)
                rand_init(pm, 0); secs = finetune(pm, tok, templates[key], Ltxt, Ly, K, ids, STEPS, lr, 0); Pq, Pc = eval_cal_test(t, templates[key], K)
                row("random LoRA init + fine-tune (same prompt)", t, 10, 0, key, Pq, Pc, params=NP, train_secs=secs, steps=STEPS, lr=lr)
    except StopIteration: pass
    except Exception: set_status("FAILED_RUN", traceback.format_exc()[-1500:])
    set_status("timing", f"cells done {len(cell_secs)}, mean cell {np.mean(cell_secs) if cell_secs else 0:.0f}s, total {time.time()-t_start:.0f}s")
    log("paw done", OUT)
