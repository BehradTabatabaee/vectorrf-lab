"""Malformed DSP requests must fail cleanly across every calculation API."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from vectorrf.app import app


TX = {"waveform": "tone", "sample_rate": 100_000, "duration": 0.002}
ROUTES = [
    ("/api/tx/plan", TX, ("sample_rate",)),
    ("/api/tx/preview", TX, ("sample_rate",)),
    ("/api/spectrum", {"tx": TX}, ("tx", "sample_rate")),
    ("/api/blind/generate", TX, ("sample_rate",)),
    ("/api/blind/analyze", {"tx": TX}, ("tx", "sample_rate")),
    ("/api/rx/loopback", {"tx": TX}, ("tx", "sample_rate")),
    ("/api/measurements", {"tx": TX}, ("tx", "sample_rate")),
    ("/api/studio/radar", {}, ("sample_rate",)),
    ("/api/studio/pipeline", {"tx": TX}, ("tx", "sample_rate")),
    ("/api/studio/ofdm", TX, ("sample_rate",)),
]
BAD_VALUES = ["oops", -1, 1e99, [], {}, None, "NaN", "Infinity", True]
BAD_SCALARS = ["oops", [], {}, None, "NaN", "Infinity"]


@pytest.mark.parametrize("route,base,path", ROUTES)
@pytest.mark.parametrize("value", BAD_VALUES, ids=str)
def test_bad_sampling_clocks_return_readable_422(route, base, path, value):
    body = deepcopy(base)
    target = body
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value
    response = TestClient(app, raise_server_exceptions=False).post(route, json=body)
    assert response.status_code == 422, (route, value, response.status_code, response.text[:300])
    payload = response.json()
    assert isinstance(payload["detail"], str) and payload["detail"]
    assert isinstance(payload["errors"], list) and payload["errors"]
    assert not any(fragment in response.text for fragment in
                   ("validation error for", "input_value", "pydantic.dev"))


TX_ROUTES = [case for case in ROUTES if case[0] in {
    "/api/tx/plan", "/api/tx/preview", "/api/spectrum", "/api/blind/generate",
    "/api/blind/analyze", "/api/rx/loopback", "/api/measurements",
    "/api/studio/pipeline", "/api/studio/ofdm",
}]


@pytest.mark.parametrize("route,base,path", TX_ROUTES)
@pytest.mark.parametrize("field", ["duration", "power_dbm", "center_hz"])
@pytest.mark.parametrize("value", BAD_SCALARS, ids=str)
def test_malformed_transmitter_scalars_fail_cleanly_across_observers(route, base, path, field, value):
    body = deepcopy(base)
    target = body
    for segment in path[:-1]:
        target = target[segment]
    target[field] = value
    response = TestClient(app, raise_server_exceptions=False).post(route, json=body)
    assert response.status_code == 422, (route, field, value, response.status_code, response.text[:300])
    payload = response.json()
    assert isinstance(payload["detail"], str) and isinstance(payload["errors"], list)
