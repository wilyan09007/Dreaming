"""frames.json + voice.m4a -> dreaming-demo.mp4 (30 fps CFR, H.264 CRF 18, AAC)."""
import json, subprocess

tl = json.load(open("timeline.json"))
fr = json.load(open("frames.json"))
total = tl["total"]
lines = []
for i, f in enumerate(fr):
    t = 0.0 if i == 0 else f["t"]
    nxt = fr[i + 1]["t"] if i + 1 < len(fr) else total
    lines.append(f"file '{f['f']}'\nduration {max(0.001, nxt - t):.4f}")
lines.append(f"file '{fr[-1]['f']}'")
open("frames.txt", "w").write("\n".join(lines) + "\n")
subprocess.run(["C:/ffmpeg/bin/ffmpeg.exe", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "frames.txt", "-i", "voice.m4a",
                "-vf", "fps=30,scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "copy", "-t", f"{total}", "-movflags", "+faststart", "dreaming-demo.mp4"], check=True)
print("ok", total)
