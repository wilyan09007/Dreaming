"""S24 arm 4: symbolic student. An LLM writes classify(text) -> label | None from the label descriptions and the k real examples per label, using regexes / keyword rules only.
The code runs in a subprocess (timeout, memory cap, scrubbed env, per-text alarm). confidence = 1 if a rule fired (returned a label) else 0, so the certificate is over fired rules."""
import ast, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path
from lib import *

CONTEXT = {"clinc": "short utterances to a voice assistant; the label is the user's intent (domain: {dom})", "banking": "messages from customers of an online bank to its support chat; the label is the customer's intent"}
RUNNER = r'''
import sys, json, time, importlib.util
try:
    import resource; resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
except Exception: pass
import signal
HAS = hasattr(signal, "setitimer")
def _to(*a): raise TimeoutError()
if HAS: signal.signal(signal.SIGALRM, _to)
spec = importlib.util.spec_from_file_location("rules", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
texts = json.load(open(sys.argv[2])); out = []; errs = 0; t0 = time.time()
for t in texts:
    try:
        if HAS: signal.setitimer(signal.ITIMER_REAL, 0.5)
        r = m.classify(t)
        if HAS: signal.setitimer(signal.ITIMER_REAL, 0)
        out.append(r if (r is None or isinstance(r, str)) else str(r))
    except BaseException:
        errs += 1; out.append(None)
        if HAS: signal.setitimer(signal.ITIMER_REAL, 0)
json.dump(dict(out=out, secs=time.time() - t0, errs=errs), open(sys.argv[3], "w"))
'''
OK_IMPORTS = {"re", "typing", "__future__"}
BAD = re.compile(r"\b(open|exec|eval|__import__|globals|locals|getattr|setattr|input|breakpoint)\s*\(|\b(os|sys|subprocess|socket|shutil|pathlib|importlib|ctypes)\s*\.")

def build_prompt(e, idx):
    ys = e["ys"][idx]; labs = e["labels"]; dom = e["name"].split("/")[1]
    ex = "\n".join(f"{labs[c]}:\n" + "\n".join(f"  - {e['ts'][i]!r}" for i, y in zip(idx, ys) if y == c) for c in range(len(labs)))
    return (f"You are writing a tiny rule-based text classifier (no machine learning).\nTask: {CONTEXT[e['family']].format(dom=dom)}.\n\nLabels (return exactly one of these strings):\n"
            + "\n".join(f"- {l}: {d}" for l, d in zip(labs, e["desc_t"])) + f"\n\nLabelled examples (k={len(idx) // len(labs)} per label):\n{ex}\n\n"
            "Write a single Python function `classify(text)` that returns exactly one of the label strings above, or None when no rule matches (prefer None over guessing).\n"
            "Rules: `import re` is the only allowed import; use only regular expressions and keyword tests on the lower-cased text; deterministic, fast, no nested quantifiers "
            "(avoid catastrophic backtracking); no file, network or system access. Reply with one ```python code block and nothing else.")

def extract_code(txt):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", txt, re.S)
    code = m.group(1) if m else (txt if "def classify" in txt else "")
    return code.strip()

def validate(code):
    if "def classify" not in code: return "no_function"
    try: tree = ast.parse(code)
    except SyntaxError: return "syntax_error"
    for n in ast.walk(tree):
        mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else ([n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module else [])
        if any(m not in OK_IMPORTS for m in mods): return "forbidden_import"
    return "forbidden_call" if BAD.search(code) else "ok"

def run_rules(code, texts, timeout=120):
    """Executes classify() on texts in a subprocess. Returns (list of str|None, seconds in the loop, per-text errors/timeouts, status)."""
    with tempfile.TemporaryDirectory() as td:
        td_ = Path(td); (td_ / "rules.py").write_text("from __future__ import annotations\n" + code, encoding="utf-8"); (td_ / "runner.py").write_text(RUNNER); (td_ / "in.json").write_text(json.dumps(texts), encoding="utf-8"); env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            p = subprocess.run([sys.executable, "-I", "runner.py", "rules.py", "in.json", "out.json"], cwd=td, env=env, timeout=timeout, capture_output=True, text=True)
            if p.returncode != 0 or not os.path.exists(os.path.join(td, "out.json")): return [None] * len(texts), 0.0, len(texts), "crash: " + p.stderr[-200:].replace("\n", " ")
            r = json.loads((td_ / "out.json").read_text()); return r["out"], r["secs"], r["errs"], "ok"
        except subprocess.TimeoutExpired: return [None] * len(texts), float(timeout), len(texts), "timeout"

def job(e, k, seed, stub=None):
    """One (task, k, draw) cell -> record fields. `stub` (tiny mode only) supplies code without calling the LLM."""
    K = e["K"]; idx = draw_support(e["ys"], k, seed, K); r = None
    if stub is None: r = llm(build_prompt(e, idx), max_tokens=6000, temperature=0.2, tag="s24-sym")
    base = dict(arm="4 symbolic rules (LLM-written)", task=e["name"], family=e["family"], k=k, seed=seed, K=K, params=0)
    if stub is None and r is None: return dict(base, status="no_llm_response_or_cost_cap")
    code = extract_code(r["text"]) if stub is None else stub
    status = validate(code); lab2i = {l: i for i, l in enumerate(e["labels"])}; lab2i |= {l.lower(): i for i, l in enumerate(e["labels"])}
    texts = e["tc"] + e["tq"]; nc = len(e["tc"])
    if status == "ok": out, secs, errs, st2 = run_rules(code, texts); status = st2
    else: out, secs, errs = [None] * len(texts), 0.0, 0
    ix = np.array([-1 if o is None else lab2i.get(o, lab2i.get(str(o).lower(), -2)) for o in out])  # -1 abstain, -2 invalid label string
    fired = (ix != -1).astype(float); ok = ix == np.concatenate([e["yc"], e["yq"]])
    sc = score_conf(fired[:nc], ok[:nc], fired[nc:], ok[nc:]); fq = fired[nc:] > 0
    return dict(base, **sc, fit_s=(r or {}).get("secs", 0.0), pred_ms_1000=1e6 * secs / max(len(texts), 1), status=status, abstain=float(1 - fq.mean()),
                acc_fired=float(ok[nc:][fq].mean()) if fq.any() else None, code_chars=len(code), n_regex=len(re.findall(r"re\.(?:search|match|fullmatch|findall|compile|sub)", code)),
                text_errs=errs, llm_cost=(r or {}).get("cost", 0.0))
