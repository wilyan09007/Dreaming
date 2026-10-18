"""The developer-view page (apps/web) against the real repo files, served by the API's static mount.

test_api.py proves the mount with a fake repo layout. These tests use the real apps/web: the page and its scripts are served,
the API still wins over the mount, the JS type-checks against the generated schema, the generated schema is current, and every
request app.js makes names a route the API really has (rename a route and this file fails, not the browser).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
WEB = REPO / "apps" / "web"
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")
LIVE_JS = (WEB / "live.js").read_text(encoding="utf-8")


def served(client: TestClient, path: str) -> str:
    r = client.get(path)
    assert r.status_code == 200, f"{path}: {r.status_code}"
    return r.text


# ------------------------------------------------------------------ served


def test_the_root_is_the_page(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'src="./app.js"' in r.text and "<script" in r.text
    assert r.text == (WEB / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["app.js", "live.js", "schema.d.ts"])
def test_the_scripts_and_types_the_page_imports_are_served(client: TestClient, name: str) -> None:
    r = client.get(f"/{name}")
    assert r.status_code == 200
    assert r.text == (WEB / name).read_text(encoding="utf-8")
    if name.endswith(".js"):
        assert "javascript" in r.headers["content-type"]  # a module script with another MIME type does not run


def test_the_launch_deck_and_every_script_it_loads_are_served(client: TestClient) -> None:
    page = served(client, "/launch/")
    scripts = re.findall(r'<script src="([^"]+)"', page)
    assert "deck.js" in scripts and len(scripts) >= 9  # numbers, deck, seven screens
    for src in scripts:
        r = client.get(f"/launch/{src}")
        assert r.status_code == 200 and r.content == (WEB / "launch" / src).read_bytes(), src


def test_the_api_routes_still_win_over_the_mount(client: TestClient) -> None:
    health = client.get("/healthz")
    assert health.headers["content-type"].startswith("application/json") and health.json()["component"] == "api"
    state = client.get("/state")
    assert state.headers["content-type"].startswith("application/json") and "sites" in state.json()
    assert client.get("/openapi.json").headers["content-type"].startswith("application/json")


# ------------------------------------------------------------------ types


@pytest.mark.skipif(shutil.which("npx") is None, reason="npx is not on PATH")
def test_app_js_and_live_js_type_check_against_the_generated_schema() -> None:
    # `npx typescript@5 tsc` cannot pick one of the package's two bins; `-p` names the package and `tsc` the command
    npx = shutil.which("npx") or "npx"  # on Windows the executable is npx.cmd; which() resolves it
    cmd = [npx, "--yes", "-p", "typescript@5", "tsc", "--noEmit", "--allowJs", "--checkJs", "--target", "es2022", "--module", "es2022",
           "--moduleResolution", "bundler", "apps/web/app.js", "apps/web/live.js"]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=180, check=False)
    out = p.stdout + p.stderr
    if p.returncode != 0 and "error TS" not in out:
        pytest.skip(f"tsc could not be fetched or started (offline?): {out.strip()[-300:]}")  # a type error always prints `error TS`
    assert p.returncode == 0, out


def test_schema_json_is_what_just_schema_writes() -> None:
    """Runs the python of the justfile's `schema` recipe, so a change to contracts.py that nobody regenerated fails here."""
    m = re.search(r'-c "([^"]+)" > apps/web/schema\.json', (REPO / "justfile").read_text(encoding="utf-8"))
    assert m, "the justfile's schema recipe no longer writes apps/web/schema.json from `python -c`"
    p = subprocess.run([sys.executable, "-c", m.group(1)], cwd=REPO, capture_output=True, text=True, timeout=60, check=True)
    assert json.loads(p.stdout) == json.loads((WEB / "schema.json").read_text(encoding="utf-8")), "contracts.py changed: run `just schema`"


# ------------------------------------------------------------------ requests


def norm(path: str) -> str:
    """'/tools/${id}/outage?x=1' and '/tools/{site}/outage' both become '/tools/{}/outage'."""
    return re.sub(r"(\$)?\{[^}]*\}", "{}", path.split("?")[0])


def requests_in(js: str, pattern: str) -> set[tuple[str, str]]:
    return {(method or "GET", norm(path)) for method, path in re.findall(pattern, js)}


def test_every_request_the_page_makes_is_a_route_the_api_has(client: TestClient) -> None:
    assert APP_JS.count("fetch(") == 1, "app.js sends every request through req(), so this test sees every URL"
    used = requests_in(APP_JS, r"""\b(?:req|cmd)\(\s*'(GET|POST)'\s*,\s*[`'](/[^`']*)[`']""")
    used |= requests_in(LIVE_JS, r"""()fetch\(`\$\{apiBase\}(/[^`]*)`""")  # live.js backfills and polls /events itself
    paths = client.get("/openapi.json").json()["paths"]
    routes = {(method.upper(), norm(path)) for path, ops in paths.items() for method in ops}
    assert used <= routes, f"routes the page calls that the API does not have: {sorted(used - routes)}"
    assert {("GET", "/healthz"), ("GET", "/state"), ("GET", "/events"), ("GET", "/benchmark")} <= used  # a parse that found nothing must not pass
    assert {p for m, p in used if m == "POST"} == {"/batch", "/batch/pause", "/batch/resume", "/budget", "/tools/{}/outage", "/replay/{}"}
