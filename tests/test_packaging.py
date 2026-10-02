from importlib import resources

from fastapi.testclient import TestClient

from vectorrf.app import app


EXPECTED_WEB_ASSETS = {
    "index.html",
    "help.html",
    "styles.css",
    "app.js",
    "studio.js",
    "workbench.js",
}


def test_package_contains_browser_assets():
    web = resources.files("vectorrf").joinpath("web")
    names = {item.name for item in web.iterdir() if item.is_file()}
    assert EXPECTED_WEB_ASSETS <= names


def test_application_serves_packaged_browser_assets():
    client = TestClient(app)
    for path in ("/", "/help", "/assets/app.js", "/assets/styles.css"):
        response = client.get(path)
        assert response.status_code == 200, path
