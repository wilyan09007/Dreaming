#!/bin/bash
# usage: launch_shard.sh <tag> "<tasks>" <max-min> [log-dir]; tries GPU types in order until a pod starts (supply errors are common)
TAG=$1; TASKS=$2; MAXMIN=$3; LOGD=${4:-.}
PY="C:/Users/<user>/Desktop/personalprojects/Dreaming/.venv/Scripts/python.exe"
cd "C:/Users/<user>/Desktop/personalprojects/Dreaming/smokes/s8_gpu_student"
for round in 1 2 3 4 5 6; do
  for GPU in "NVIDIA GeForce RTX 4090" "NVIDIA RTX A6000" "NVIDIA GeForce RTX 3090 Ti" "NVIDIA GeForce RTX 3090"; do
    "$PY" -u launch_runpod_s22.py --script s22_arch.py --args "--tasks $TASKS --out results.json" --tag "$TAG" --max-min "$MAXMIN" --gpu "$GPU" --cuda "${CUDA:-12.8}" > "$LOGD/launch$TAG.txt" 2>&1 &
    PID=$!; sleep 25
    if kill -0 $PID 2>/dev/null && grep -q "^pod " "$LOGD/launch$TAG.txt"; then echo "started $TAG on $GPU"; wait $PID; exit 0; fi
    kill $PID 2>/dev/null; echo "no pod for $TAG on $GPU ($(grep -o 'SUPPLY_CONSTRAINT' "$LOGD/launch$TAG.txt" | head -1))"
  done
  sleep 60
done
echo "gave up on $TAG"
