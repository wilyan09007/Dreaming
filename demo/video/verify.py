"""Contact sheets of dreaming-demo.mp4 at every phrase midpoint (plus intro/outro), for a visual check.

Writes build/sheet-N.png (3x3 tiles, each labelled with time and shot). Run from demo/video: python verify.py [mp4]
"""
import json, os, subprocess, sys

FF = os.environ.get("FFMPEG") or ("C:/ffmpeg/bin/ffmpeg.exe" if os.path.exists("C:/ffmpeg/bin/ffmpeg.exe") else "ffmpeg")
mp4 = sys.argv[1] if len(sys.argv) > 1 else "dreaming-demo.mp4"
tl = json.load(open("timeline.json"))
pts = [(1.8, "intro")] + [((p["start"] + p["end"]) / 2, p["shot"]) for p in tl["phrases"]] + [(tl["total"] - 0.5, "outro")]
os.makedirs("build/v", exist_ok=True)
for i, (t, name) in enumerate(pts):
    label = f"{t:6.2f}s {name}".replace(":", r"\:")
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", mp4, "-frames:v", "1", "-vf",
                    f"scale=640:360,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='{label}':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6",
                    f"build/v/{i:02d}.png"], check=True)
for k in range(0, len(pts), 9):
    n = min(9, len(pts) - k)
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-start_number", str(k), "-i", "build/v/%02d.png",
                    "-frames:v", "1", "-vf", f"tile=3x3:nb_frames={n}:padding=4", f"build/sheet-{k // 9}.png"], check=True)
print("sheets:", [f"build/sheet-{k // 9}.png" for k in range(0, len(pts), 9)])
