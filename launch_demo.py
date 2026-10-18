"""Launch the Dreaming demo: the backends, then one chromeless window per demo module.

    python launch_demo.py                 # all seven modules + the live app, local backends (works offline)
    python launch_demo.py race bench psi  # only these modules
    python launch_demo.py --url https://dreaming-demo.fly.dev   # the live app from the deployed stack, no local API
    python launch_demo.py --check         # start the backends, fetch every page, exit (no windows: CI, cloud sessions)

Modules: hook race bench psi train build close. Each window shows one module (launch/?only=<id>), replays when it is
brought to the front, R replays it, F toggles fullscreen. On Windows the modules open maximized on the external display
and the live app on the laptop's (primary) one; --swap flips that. Close every window or press Ctrl+C to stop.
Standard library only: plain `python` works; the local API needs `uv` (https://docs.astral.sh/uv/).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "dreaming" / "apps" / "web"
MODULES = ["hook", "race", "bench", "psi", "train", "build", "close"]
OFFLINE = {"DATABASE_URL": "", "TEACHER": "rule", "ENCODER": "char", "TRAINER": "local"}  # the API with no accounts (justfile `dev`)
LIVE_TITLE = "Dreaming"  # apps/web/index.html <title>


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *_: object) -> None:
        pass


def static_server() -> str:
    """apps/web from a thread on a free port: the modules are static files, so their windows never wait for the API."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(WEB)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
        except OSError:
            s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def spawn(cmd: list[str], **kw: object) -> subprocess.Popen[bytes]:
    """A quiet child we can kill with its whole tree (uv -> python, browser -> renderers)."""
    group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
    out = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, **kw}
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **group, **out)  # type: ignore[call-overload]


def stop(p: subprocess.Popen[bytes] | None) -> None:
    if p is None or p.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        os.killpg(p.pid, signal.SIGTERM)


def start_api(port: int, log: Path) -> subprocess.Popen[bytes] | None:
    uv = shutil.which("uv")
    if not uv:
        print("  api       skipped: uv is not on PATH (https://docs.astral.sh/uv/)")
        return None
    cmd = [uv, "run", "--directory", str(ROOT / "dreaming"), "--all-packages", "python", "-m", "uvicorn",
           "dreaming_api.wiring:app", "--factory", "--host", "127.0.0.1", "--port", str(port)]
    return spawn(cmd, env={**os.environ, **OFFLINE}, stdout=log.open("wb"), stderr=subprocess.STDOUT)


def up(url: str, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except OSError:
            time.sleep(0.25)
    return False


def browser() -> str | None:
    """Chrome or Edge: their --app windows have no tabs, toolbar or address bar."""
    local = os.environ.get("LOCALAPPDATA", "")
    paths = [rf"{local}\Google\Chrome\Application\chrome.exe", r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
             *filter(None, map(shutil.which, ["google-chrome", "chromium", "chromium-browser", "microsoft-edge", "chrome", "msedge"]))]
    return next((p for p in paths if Path(p).is_file()), None)


def arrange(want: dict[str, bool], swap: bool) -> None:
    """Windows: find each window by title, maximize it on its display (True = the external one, False = the laptop's)."""
    if sys.platform != "win32":
        return  # ponytail: elsewhere the OS places the windows; add an xdotool/AppleScript pass if that ever matters
    import ctypes
    from ctypes import wintypes

    u = ctypes.windll.user32
    try:
        u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2: real pixels on every display
    except AttributeError:
        pass  # before Windows 10 1703: scaled coordinates, still on the right display
    H, I = wintypes.HANDLE, ctypes.c_int
    u.GetMonitorInfoW.argtypes = [H, ctypes.c_void_p]
    u.GetWindowTextLengthW.argtypes, u.IsWindowVisible.argtypes = [H], [H]
    u.GetWindowTextW.argtypes = [H, wintypes.LPWSTR, I]
    u.ShowWindow.argtypes, u.SetForegroundWindow.argtypes = [H, I], [H]
    u.SetWindowPos.argtypes = [H, H, I, I, I, I, wintypes.UINT]

    class Info(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    displays: list[tuple[bool, wintypes.RECT]] = []  # (primary, work area)

    def on_display(hmon: int, _dc: int, _rect: object, _arg: int) -> bool:
        info = Info(cbSize=ctypes.sizeof(Info))
        u.GetMonitorInfoW(hmon, ctypes.byref(info))
        displays.append((bool(info.dwFlags & 1), info.rcWork))
        return True

    u.EnumDisplayMonitors(None, None, ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, ctypes.c_void_p, wintypes.LPARAM)(on_display), 0)
    primary = next(r for p, r in displays if p)
    external = next((r for p, r in displays if not p), primary)  # a laptop is the primary display; the big monitor extends it
    if swap:
        primary, external = external, primary

    found: dict[str, int] = {}

    def on_window(hwnd: int, _arg: int) -> bool:
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        title = buf.value.split(" - ")[0]  # Edge appends " - Microsoft Edge"
        if title in want and u.IsWindowVisible(hwnd):
            found.setdefault(title, hwnd)
        return True

    enum = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(on_window)
    end = time.monotonic() + 20  # titles appear once each page has run its scripts
    while len(found) < len(want) and time.monotonic() < end:
        u.EnumWindows(enum, 0)
        time.sleep(0.2)
    for title in reversed(list(want)):  # placed last = on top, so the first module ends up in front
        hwnd, r = found.get(title), external if want[title] else primary
        if hwnd is None:
            print(f"  window    {title!r} did not appear; it stays where the browser opened it")
            continue
        u.ShowWindow(hwnd, 9)  # restore, so it can move
        u.SetWindowPos(hwnd, 0, r.left, r.top, r.right - r.left, r.bottom - r.top, 0x0040)
        u.ShowWindow(hwnd, 3)  # maximize on that display
        for z in (-1, -2):  # topmost, then not: raises it over the terminal, which HWND_TOP from a background process can't
            u.SetWindowPos(hwnd, z, 0, 0, 0, 0, 0x0003)
    if first := found.get(next(iter(want))):
        u.keybd_event(0x12, 0, 0, 0)  # an Alt tap lets this process take the foreground (focus-stealing rule)
        u.keybd_event(0x12, 0, 2, 0)
        u.SetForegroundWindow(first)  # the clicker's keys go to the first module
    print(f"  windows   {len(found)}/{len(want)} placed ({len(displays)} display{'s' * (len(displays) > 1)})")


def check(base: str, live: str | None) -> bool:
    """Fetch every module page, every script the deck loads, and the live app: what a window would need."""
    page = urllib.request.urlopen(f"{base}/launch/", timeout=5).read().decode()
    urls = [f"{base}/launch/?only={m}" for m in MODULES] + [f"{base}/launch/{s}" for s in re.findall(r'<script src="([^"]+)"', page)]
    urls += [f"{live}/healthz", f"{live}/"] if live else []
    bad = [u for u in urls if not up(u, 5)]
    print(f"  check     {len(urls) - len(bad)}/{len(urls)} pages answered" + (": missing " + ", ".join(bad) if bad else ""))
    return not bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="*", metavar="module", help=" ".join(MODULES) + " (default: all)")
    ap.add_argument("--url", help="use this deployed stack for the live app instead of starting the local API")
    ap.add_argument("--no-live", action="store_true", help="modules only: no live app window, no API")
    ap.add_argument("--swap", action="store_true", help="modules on the primary display, the live app on the other")
    ap.add_argument("--check", action="store_true", help="start the backends, fetch every page, exit")
    a = ap.parse_args()
    if unknown := [m for m in a.modules if m not in MODULES]:
        ap.error(f"unknown module {', '.join(unknown)}; pick from {' '.join(MODULES)}")
    modules = a.modules or MODULES

    t0 = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="dreaming-demo-"))  # a fresh browser profile each run: no stale cached JS, windows we own
    api = live = main_browser = None
    if not a.no_live and not a.url:
        port = free_port(8000)
        api, live = start_api(port, tmp / "api.log"), f"http://127.0.0.1:{port}"
    elif a.url:
        live = a.url.rstrip("/")
    base = static_server()
    print(f"  modules   {base}/launch/  ({len(modules)} module{'s' * (len(modules) > 1)})")
    try:
        exe = None if a.check else browser()
        flags = [exe, f"--user-data-dir={tmp / 'browser'}", "--no-first-run", "--no-default-browser-check", "--new-window"]
        urls = {f"{m} · dreaming": f"{base}/launch/?only={m}" for m in modules}
        if exe:  # the module windows open now; only the live app waits for the API
            main_browser = spawn([*flags, f"--app={next(iter(urls.values()))}"])
            lock = [tmp / "browser" / n for n in ("lockfile", "SingletonLock")]
            end = time.monotonic() + 10
            while not any(p.exists() or p.is_symlink() for p in lock) and time.monotonic() < end:
                time.sleep(0.05)  # once the first browser owns the profile, the rest open as its windows and exit
            for url in list(urls.values())[1:]:
                spawn([*flags, f"--app={url}"])
        api_down = bool(live and api is not None and not up(f"{live}/healthz", 90))
        if api_down:
            log = Path(tempfile.gettempdir()) / "dreaming-demo-api.log"  # tmp is removed on exit; keep the log readable
            shutil.copyfile(tmp / "api.log", log)
            print(f"  api       did not come up; log: {log} (first run downloads torch: rerun once uv has synced)")
            live = None
        elif live:
            print(f"  live app  {live}/  (up in {time.monotonic() - t0:.1f} s)")
        if a.check:
            return 0 if check(base, live) and not api_down else 1
        if not exe:
            print("  no Chrome or Edge found; open these yourself:", *urls.values(), live or "", sep="\n    ")
            input("  Enter stops the backends ")
            return 0
        if live:
            urls[LIVE_TITLE] = f"{live}/"
            spawn([*flags, f"--app={live}/"])
        arrange({t: not t == LIVE_TITLE for t in urls}, a.swap)
        print(f"  ready in {time.monotonic() - t0:.1f} s. Close every window or press Ctrl+C to stop.")
        while main_browser.poll() is None:
            time.sleep(0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        stop(api)
        stop(main_browser)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
