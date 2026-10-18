"""Voiceover edit: tighten pauses, drop the part1 false start, speed up, normalize.

Writes voice.m4a and timeline.json (final-timeline times of phrases and subtitle cues).
Run from demo/video: python audio.py
"""
import json, re, subprocess

INTRO = 2.5          # logo reveal before the first word
OUTRO_HOLD = 2.6     # outro screen after "Keep on dreaming."
TEMPO = 1.17
KEEP_GAP = (0.12, 0.14)  # silence kept before/after each cut

# file, start, end of usable speech (measured with silencedetect -38dB; part1 false start cut)
TAKES = [("part1.m4a", 1.22, 98.78), ("part2.m4a", 0.87, 38.05)]

# phrase start times (file seconds) from HANDOFF.md, in shot order
PHRASES = [("part1.m4a", t) for t in (0, 10.5, 22.3, 34.6, 37.9, 46.2, 60.2, 68.8, 75.3, 83.9, 90.3)] + \
          [("part2.m4a", t) for t in (0, 9.4, 14.0, 17.9, 30.0, 36.6)]
SHOTS = ["hook", "race", "bench", "run", "psi", "bound", "budget", "replaytab", "replayrows", "hash", "replay",
         "build", "buildrace", "journal", "scores", "close", "outro"]

WEAK = {"a", "the", "of", "to", "and", "our", "with", "in", "on", "by", "that", "your", "their", "we", "from", "at", "each"}  # never end a line on these
FIX = {"SIE.": "Psi.", "SIE": "Psi", "dreaming": "Dreaming", "-language": "language", "modeling": "model a"}


def silences(f):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", f, "-af", "silencedetect=noise=-30dB:d=0.2", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    s = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", out)]
    e = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]
    return list(zip(s, e))


def gaps(f, words, lo, hi):
    """Pauses to shorten: detected silences plus breath gaps between words (whisper)."""
    g = [(a, b) for a, b in silences(f)]
    g += [(w0["e"] + 0.05, w1["s"] - 0.03) for w0, w1 in zip(words, words[1:]) if w1["s"] - w0["e"] > 0.3]
    g = sorted((max(a, lo), min(b, hi)) for a, b in g if b > lo and a < hi)
    merged = []
    for a, b in g:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a > sum(KEEP_GAP) + 0.05]


def main():
    words = json.load(open("words.json"))
    segs = []  # (file, a, b)
    for f, lo, hi in TAKES:
        cur = lo
        for a, b in gaps(f, words[f], lo, hi):
            segs.append((f, cur, a + KEEP_GAP[0]))
            cur = b - KEEP_GAP[1]
        segs.append((f, cur, hi))

    def tmap(f, t):  # file time -> final timeline
        acc = 0.0
        for sf, a, b in segs:
            if sf == f and a <= t <= b:
                return INTRO + (acc + t - a) / TEMPO
            if sf == f and t < a:
                return INTRO + acc / TEMPO
            acc += b - a
        return INTRO + acc / TEMPO

    voice_len = sum(b - a for _, a, b in segs) / TEMPO
    total = INTRO + voice_len + OUTRO_HOLD

    # ffmpeg graph: each segment trimmed + 8 ms fades, concatenated
    parts, labels = [], []
    for i, (f, a, b) in enumerate(segs):
        src = 0 if f == "part1.m4a" else 1
        d = b - a
        parts.append(f"[{src}:a]atrim={a:.3f}:{b:.3f},asetpts=PTS-STARTPTS,afade=t=in:d=0.008,"
                     f"afade=t=out:st={d - 0.008:.3f}:d=0.008[s{i}]")
        labels.append(f"[s{i}]")
    graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(segs)}:v=0:a=1,afftdn=nf=-25,highpass=f=70," \
        f"atempo={TEMPO},loudnorm=I=-16:TP=-1.5:LRA=11,adelay={int(INTRO * 1000)}:all=1,apad=whole_dur={total:.3f}[out]"
    open("audio_graph.txt", "w").write(graph)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "part1.m4a", "-i", "part2.m4a",
                    "-filter_complex_script", "audio_graph.txt", "-map", "[out]", "-ar", "48000", "-c:a", "aac",
                    "-b:a", "192k", "voice.m4a"], check=True)

    # words -> final timeline, merged tokens, text fixes
    toks = []
    for f, lo, hi in TAKES:
        for w in words[f]:
            if not (lo - 0.6 <= w["s"] < hi):
                continue
            raw = " language" if w["w"].strip() == "-language" else w["w"]
            t = raw.strip()
            if toks and toks[-1]["f"] == f and (not raw.startswith(" ") or t[0] in "%.,-"):
                toks[-1]["w"] += t
                toks[-1]["e"] = tmap(f, w["e"])
                continue
            toks.append({"f": f, "fs": w["s"], "w": FIX.get(t, t), "s": tmap(f, max(w["s"], lo)), "e": tmap(f, w["e"])})

    # phrases
    starts = [tmap(f, max(t, lo)) for (f, t), lo in zip(PHRASES, [TAKES[0][1]] * 11 + [TAKES[1][1]] * 6)]
    phrases = []
    for i, st in enumerate(starts):
        f = PHRASES[i][0]
        nxt = PHRASES[i + 1] if i + 1 < len(PHRASES) else None
        ws = [t for t in toks if t["f"] == f and t["fs"] >= PHRASES[i][1] - 0.01 and
              (nxt is None or nxt[0] != f or t["fs"] < nxt[1] - 0.01)]
        phrases.append({"shot": SHOTS[i], "start": ws[0]["s"], "end": ws[-1]["e"],
                        "text": " ".join(w["w"] for w in ws), "words": [[w["w"], round(w["s"], 3)] for w in ws]})

    # subtitle cues: split each phrase into n balanced one-line chunks (<= ~44 chars), preferring punctuation
    cues = []
    for p in phrases:
        ws = p["words"]
        L = len(p["text"])
        n = -(-L // 50)
        ends, pos = [], 0  # char position after each word
        for w, _ in ws:
            pos += len(w) + 1
            ends.append(pos)
        cuts = []
        for k in range(1, n):
            ideal = k * L / n
            cuts.append(min(range(len(ws) - 1), key=lambda j: abs(ends[j] - ideal) - (7 if ws[j][0][-1] in ",." else 0)
                            + (6 if ws[j][0].lower() in WEAK else 0)))
        prev = 0
        for c in sorted(set(cuts)) + [len(ws) - 1]:
            chunk = ws[prev:c + 1]
            cues.append({"text": " ".join(w for w, _ in chunk), "start": chunk[0][1]})
            prev = c + 1
    # a cue on screen under ~1 s is unreadable: fold it into the next line of the same phrase when that stays short
    for i in range(len(cues) - 2, -1, -1):
        a, b = cues[i], cues[i + 1]
        if b["start"] - a["start"] < 1.0 and len(a["text"]) + len(b["text"]) < 62 and b["start"] - a["start"] > 0:
            b["text"], b["start"] = a["text"] + " " + b["text"], a["start"]
            del cues[i]
    for i, c in enumerate(cues):
        nxt = cues[i + 1]["start"] if i + 1 < len(cues) else phrases[-1]["end"] + 0.4
        c["end"] = min(nxt, c["start"] + 6.0)
    for c in cues:
        c["start"], c["end"] = round(c["start"], 3), round(c["end"], 3)

    # shots: each starts a beat before its phrase; intro and outro bracket them
    shots = [{"name": "intro", "start": 0.0}]
    for p in phrases:
        shots.append({"name": p["shot"], "start": round(p["start"] - 0.2, 3)})
    for i, s in enumerate(shots):
        s["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else round(total, 3)
    json.dump({"total": round(total, 3), "tempo": TEMPO, "phrases": phrases, "cues": cues, "shots": shots},
              open("timeline.json", "w"), indent=1)
    print(f"segments {len(segs)}  voice {voice_len:.2f}s  total {total:.2f}s  cues {len(cues)}")
    for c in cues:
        print(f"{c['start']:7.2f} {c['end']:7.2f}  {c['text']}")


if __name__ == "__main__":
    main()
