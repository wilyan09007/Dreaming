"""S21 extractor (GPU on the pod). For each model: last-token and query-mean hidden states at several layers for every prompt set,
and for Banking77 sets the greedy answer + answer-token logprobs + two T=0.7 samples. Writes results.json (+ feats_<tag>.npz).
Usage: python extract.py --models Qwen/Qwen2.5-0.5B-Instruct [--require-cuda] [--limit N] [--sets b200,...] [--out results.json] [--featdir .]
--random-tiny: local pipeline check only (2-layer random Qwen2 built from the cached Qwen2.5-0.5B config; no weights downloaded)."""
import os, sys, json, time, argparse
import numpy as np, torch
p = argparse.ArgumentParser()
p.add_argument("--models", default="Qwen/Qwen2.5-0.5B-Instruct"); p.add_argument("--data", default=""); p.add_argument("--limit", type=int, default=0)
p.add_argument("--sets", default="b200,b1000tr,b1000new,rb2000"); p.add_argument("--bs", type=int, default=32); p.add_argument("--out", default="results.json")
p.add_argument("--featdir", default="."); p.add_argument("--tag", default=""); p.add_argument("--require-cuda", action="store_true"); p.add_argument("--random-tiny", action="store_true")
a = p.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(a.data or os.path.join(HERE, "data.json"), encoding="utf-8"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
if a.require_cuda and dev != "cuda": sys.exit("no CUDA on this pod: refusing to run the extraction on CPU")
dt = torch.bfloat16 if dev == "cuda" else torch.float32
print(f"device={dev} dtype={dt} torch={torch.__version__}", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers; print("transformers", transformers.__version__, flush=True)
EOS_LOOSE = None

def build(tok, kind, text):
    msgs = ([{"role": "system", "content": D["sys_prompt"]}] if kind == "b77" else []) + [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def run_model(name):
    tok = AutoTokenizer.from_pretrained(name); tok.padding_side = "left"
    if a.random_tiny:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B"); cfg.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2, layer_types=cfg.layer_types[:4] if getattr(cfg, "layer_types", None) else None))
        torch.manual_seed(0); model = AutoModelForCausalLM.from_config(cfg).to(dt).to(dev).eval()
    else: model = AutoModelForCausalLM.from_pretrained(name, dtype=dt).to(dev).eval()
    nl = model.config.num_hidden_layers; layers = sorted({max(1, round(nl * f)) for f in (.25, .4, .55, .7, .85, 1.0)})
    print(f"[{name}] layers={nl} hidden={model.config.hidden_size} saved layers {layers}", flush=True)
    out = dict(model=name, n_layers=nl, layers=layers, sets={}); feats = {}
    for sname in a.sets.split(","):
        texts = D[sname]; texts = texts[: a.limit] if a.limit else texts
        kind = "rb" if sname.startswith("rb") else "b77"
        prompts = [build(tok, kind, t) for t in texts]
        pre = tok(build(tok, kind, "").split("<|im_start|>user")[0], add_special_tokens=False)["input_ids"] if kind == "b77" else []
        n = len(prompts); lens = [len(tok(q, add_special_tokens=False)["input_ids"]) for q in prompts]
        order = np.argsort(lens)[::-1]  # long first
        last = np.zeros((n, len(layers), model.config.hidden_size), np.float16); mean = np.zeros_like(last)
        gen = [None] * n; t0 = time.time()
        for b in range(0, n, a.bs):
            ix = order[b:b + a.bs]; enc = tok([prompts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            L = enc["input_ids"].shape[1]; am = enc["attention_mask"]; pos = (am.cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                hs = model(**enc, position_ids=pos, output_hidden_states=True).hidden_states
            for k, l in enumerate(layers):
                h = hs[l].float()
                last[ix, k] = np.clip(h[:, -1].cpu().numpy(), -6e4, 6e4).astype(np.float16)
                # mean over the variable part: b77 = tokens after the shared system prefix, rb = all real tokens
                sl = torch.tensor([lens[i] - len(pre) if kind == "b77" else lens[i] for i in ix], device=dev).clamp(min=1)
                mk = (torch.arange(L, device=dev)[None, :] >= (L - sl)[:, None]).float()
                mean[ix, k] = np.clip(((h * mk[..., None]).sum(1) / mk.sum(1)[:, None]).cpu().numpy(), -6e4, 6e4).astype(np.float16)
            if kind == "b77":
                torch.manual_seed(1000 + b)
                common = dict(max_new_tokens=16, pad_token_id=tok.pad_token_id, repetition_penalty=1.0, return_dict_in_generate=True, output_logits=True)
                g = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, **common)
                s = model.generate(**enc, do_sample=True, temperature=0.7, top_p=1.0, top_k=0, num_return_sequences=2, max_new_tokens=16, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
                eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
                seqs = g.sequences[:, L:]; lg = torch.stack(g.logits, 1).float().log_softmax(-1)  # [B, T, V]
                top2 = lg.topk(2, -1).values; ch = lg.gather(-1, seqs[..., None])[..., 0]
                for r, i in enumerate(ix):
                    toks = seqs[r].tolist(); T = next((t for t, x in enumerate(toks) if x in eos), len(toks))  # number of answer tokens before EOS
                    gen[i] = dict(text=tok.decode(toks[:T], skip_special_tokens=True), lp=ch[r, :T].tolist(), gap=(top2[r, :T, 0] - top2[r, :T, 1]).tolist(),
                                  eos_lp=float(ch[r, T]) if T < len(toks) else None, eos_gap=float(top2[r, T, 0] - top2[r, T, 1]) if T < len(toks) else None,
                                  top1_lp_first=float(top2[r, 0, 0]), samples=[tok.decode(s[2 * r + j, L:], skip_special_tokens=True) for j in range(2)])
            if (b // a.bs) % 10 == 0: print(f"  [{name.split('/')[-1]}] {sname} {b + len(ix)}/{n} {time.time() - t0:.0f}s", flush=True)
        el = time.time() - t0
        out["sets"][sname] = dict(n=n, seconds=el, gen=gen, mean_prompt_tokens=float(np.mean(lens)))
        feats[f"{sname}_last"] = last; feats[f"{sname}_mean"] = mean
        print(f"[{name}] {sname}: n={n} {el:.1f}s ({n / el:.1f} prompts/s) mean prompt tokens {np.mean(lens):.0f}", flush=True)
        # incremental save: a vanished pod loses at most one set
        json.dump([out], open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
        np.savez(os.path.join(a.featdir, f"feats_{name.split('/')[-1] + a.tag}.npz"), **feats)
    if dev == "cuda":  # batch-1 latency of the full greedy answer on 20 Banking77 prompts, for the cost line
        ps = [build(tok, "b77", t) for t in D["b200"][:20]]; ts = []
        for q in ps:
            e = tok(q, return_tensors="pt", add_special_tokens=False).to(dev); torch.cuda.synchronize(); t0 = time.time()
            model.generate(**e, do_sample=False, max_new_tokens=16, pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None); torch.cuda.synchronize(); ts.append(time.time() - t0)
        out["batch1_latency_ms"] = float(np.median(ts[3:]) * 1000)
    tag = name.split("/")[-1] + a.tag
    np.savez(os.path.join(a.featdir, f"feats_{tag}.npz"), **feats)
    del model; torch.cuda.empty_cache() if dev == "cuda" else None
    return out

res = [run_model(m) for m in a.models.split(",")]
json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
print("wrote", a.out, flush=True)
