"""Cut the two voiceover takes into one tight voice track and write the final-time timeline.

Usage: python build_audio.py <dir with part1.m4a part2.m4a> <words.json>
Writes voice.m4a and timeline.json in the cwd.
"""
import json, subprocess, sys

SRC, WORDS = sys.argv[1], sys.argv[2]
INTRO, OUTRO, TARGET = 2.6, 2.8, 116.0  # seconds; TARGET = whole video
PHRASE_GAP, PAUSE_CAP, PRE, POST = 0.45, 0.26, 0.06, 0.16
BREATH = {4: 0.25, 8: 0.25, 12: 0.2, 14: 0.2, 16: 0.2, 17: 0.35}  # extra room before these phrases (scene changes)
FIX = {"SIE.": "Psi.", "SIE": "Psi", "-language": "language", "modeling": "model a", "dreaming": "Dreaming", "dreaming.": "dreaming."}

d = json.load(open(WORDS))
takes = [("part1.m4a", d.get("part1.m4a") or d["part1.wav"]), ("part2.m4a", d.get("part2.m4a") or d["part2.wav"])]

phrases = []  # (input index, [words])
for i, (_, ws) in enumerate(takes):
    cur = []
    for w in ws:
        if cur and w["s"] - cur[-1]["e"] > PHRASE_GAP:
            phrases.append((i, cur)); cur = []
        cur.append(w)
    phrases.append((i, cur))
phrases = [p for p in phrases if not (p[0] == 0 and p[1][0]["s"] > 99)]  # the false start at the end of part 1
phrases[-1][1][-1]["w"] = " dreaming."
assert len(phrases) == 17, len(phrases)

segs, t = [], 0.0  # (input, a, b) kept in order; t = position in the concatenated (pre-tempo) track
timeline = []
for n, (inp, ws) in enumerate(phrases, start=1):
    if n in BREATH:
        segs.append((None, BREATH[n])); t += BREATH[n]
    a = ws[0]["s"] - PRE
    pw = []
    for k, w in enumerate(ws):
        if k and w["s"] - ws[k - 1]["e"] > PAUSE_CAP:  # long pause inside a phrase: keep only PAUSE_CAP of it
            b = ws[k - 1]["e"] + PAUSE_CAP / 2
            segs.append((inp, a, b)); t += b - a
            a = w["s"] - PAUSE_CAP / 2
        pos = t + (w["s"] - a)
        txt = w["w"].strip()
        pw.append({"w": FIX.get(txt, txt), "t": pos, "e": t + (w["e"] - a)})
    b = ws[-1]["e"] + POST
    segs.append((inp, a, b)); t += b - a
    timeline.append({"n": n, "words": pw})

tempo = min(1.22, max(1.0, t / (TARGET - INTRO - OUTRO)))
for p in timeline:
    for w in p["words"]:
        w["t"] = round(INTRO + w["t"] / tempo, 3); w["e"] = round(INTRO + w["e"] / tempo, 3)
    p["start"], p["end"] = p["words"][0]["t"], p["words"][-1]["e"]
total = round(INTRO + t / tempo + OUTRO, 3)

parts, labels = [], []
for k, s in enumerate(segs):
    if s[0] is None:
        parts.append(f"anullsrc=r=48000:cl=mono,atrim=0:{s[1]:.3f}[g{k}]")
    else:
        parts.append(f"[{s[0]}:a]atrim={s[1]:.3f}:{s[2]:.3f},asetpts=PTS-STARTPTS,aformat=sample_rates=48000:channel_layouts=mono[g{k}]")
    labels.append(f"[g{k}]")
ms = int(INTRO * 1000)
parts.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1,afftdn=nf=-28,atempo={tempo:.4f},"
             f"loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000,adelay={ms},apad,atrim=0:{total}[out]")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{SRC}/part1.m4a", "-i", f"{SRC}/part2.m4a",
                "-filter_complex", ";".join(parts), "-map", "[out]", "-ac", "2", "-c:a", "aac", "-b:a", "192k", "voice.m4a"], check=True)
json.dump({"intro": INTRO, "total": total, "tempo": round(tempo, 4), "phrases": timeline}, open("timeline.json", "w"), indent=1)
print(f"tempo {tempo:.3f}  total {total:.2f}s")
for p in timeline:
    print(f"{p['n']:2d} {p['start']:7.2f}-{p['end']:7.2f} {' '.join(w['w'] for w in p['words'])[:90]}")
