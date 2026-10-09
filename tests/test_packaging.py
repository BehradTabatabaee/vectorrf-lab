import re
from importlib import resources
from pathlib import Path

from fastapi.testclient import TestClient

from vectorrf.app import app

WEB = Path(resources.files("vectorrf").joinpath("web"))
IMPORT = re.compile(r"""(?:import|export)\s[^'"]*?from\s+['"](\.{1,2}/[^'"]+)['"]|import\(['"](\.{1,2}/[^'"]+)['"]\)""")


def modules():
    return sorted((WEB / "js").rglob("*.js"))


def test_package_contains_browser_assets():
    names = {p.relative_to(WEB).as_posix() for p in WEB.rglob("*") if p.is_file()}
    assert {"index.html", "help.html", "styles.css", "js/main.js"} <= names
    assert len(modules()) >= 20


def test_every_module_import_resolves():
    """ES modules fail at load time on a wrong relative path; catch it without a browser."""
    for module in modules():
        for groups in IMPORT.findall(module.read_text(encoding="utf-8")):
            target = next(g for g in groups if g)
            assert (module.parent / target).resolve().is_file(), f"{module.relative_to(WEB)} imports missing {target}"


def test_index_loads_the_module_entry_point():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="/assets/js/main.js">' in html


def test_application_serves_packaged_browser_assets():
    client = TestClient(app)
    for path in ("/", "/help", "/assets/styles.css"):
        assert client.get(path).status_code == 200, path
    for module in modules():
        path = "/assets/" + module.relative_to(WEB).as_posix()
        response = client.get(path)
        assert response.status_code == 200, path
        # Browsers refuse to execute modules served with a non-JavaScript MIME type.
        assert "javascript" in response.headers["content-type"], (path, response.headers["content-type"])


def test_ui_assets_are_revalidated_after_upgrades():
    """Heuristic browser caching could otherwise mix modules from two releases."""
    client = TestClient(app)
    for path in ("/", "/help", "/assets/js/main.js", "/assets/js/core/state.js"):
        assert client.get(path).headers["cache-control"] == "no-cache", path
    assert "cache-control" not in client.get("/api/health").headers
