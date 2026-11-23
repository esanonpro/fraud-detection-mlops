from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)

PAYLOAD = {"Time": 0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": 10.0}


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "online"


def test_predict_returns_503_or_prediction():
    r = client.post("/predict", json=PAYLOAD)
    if client.get("/").json()["model_loaded"]:
        assert r.status_code == 200
        body = r.json()
        assert body["is_fraud"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0
    else:
        assert r.status_code == 503


def test_predict_rejects_invalid_payload():
    r = client.post("/predict", json={"Time": 0})
    assert r.status_code == 422
