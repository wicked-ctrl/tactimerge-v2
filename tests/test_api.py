# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    """GET /health should return 200 and include the api_url."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "api_url" in body
    assert body["status"] == "ok"

def test_root():
    """GET / should return the welcome message."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to TactiMerge API"}

def test_predict():
    """POST /predict with valid CSV data should return a score and xG."""
    payload = {
        "home_team":   "Arsenal",
        "away_team":   "Liverpool",
        "era":         "2022-23",
        "injuries":    [],
        "new_signings": []
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    # check keys and basic types
    assert set(body.keys()) == {"home_team", "away_team", "predicted_score", "expected_xg"}
    assert body["home_team"] == "Arsenal"
    assert isinstance(body["predicted_score"], str)
    assert isinstance(body["expected_xg"], float)

@pytest.mark.vcr  # optional: record and replay OpenAI calls if you use pytest-vcr
def test_analyze():
    """POST /analyze should return a non-empty playstyle_summary."""
    payload = {
        "team":         "Arsenal",
        "era":          "2022-23",
        "injuries":     [],
        "new_signings": []
    }
    r = client.post("/analyze", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"team", "era", "playstyle_summary"}
    assert body["team"] == "Arsenal"
    # We expect some text back from OpenAI:
    assert isinstance(body["playstyle_summary"], str)
    assert len(body["playstyle_summary"].strip()) > 0
