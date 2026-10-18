# pyright: basic
"""Modal smoke: the deployed `embed` function returns a 384-float vector, twice, byte for byte the same.

Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET and `just deploy-trainer` done once. Prints one line per check; exits 1 on failure.
The first call includes the container cold start (image pull, encoder load), the second is warm.
"""

from __future__ import annotations

import sys
import time
from array import array
from typing import Any

import modal

DIM = 384  # BAAI/bge-small-en-v1.5
failed: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail != "" else ""))
    if not ok:
        failed.append(name)


def call(fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn.remote(["hello"])
    return out, time.perf_counter() - t0


def main() -> int:
    try:
        fn = modal.Function.from_name("dreaming-trainer", "embed")
        first, cold = call(fn)
        second, warm = call(fn)
    except modal.exception.Error as e:
        check("embed is deployed and reachable", False, f"{type(e).__name__}: {e}")
        return 1
    vec = first[0] if isinstance(first, list) and first else []
    check("embed(['hello']) returns one vector of 384 floats", len(first) == 1 and len(vec) == DIM and all(isinstance(x, float) for x in vec), f"length {len(vec)}")
    same = isinstance(second, list) and bool(second) and array("d", vec).tobytes() == array("d", second[0]).tobytes()
    check("second call is byte-equal to the first (determinism)", same)
    check("timing", True, f"cold {cold:.2f}s, warm {warm:.2f}s")
    image_id = first.get("MODAL_IMAGE_ID") if isinstance(first, dict) else None  # embed returns bare vectors today
    check("MODAL_IMAGE_ID", True, image_id or "not returned by embed (vectors only); the image id is recorded in env_sha by train()")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
