"""Stage 3: end-to-end API flow in self-contained demo mode."""

import pathlib
import sys

from fastapi.testclient import TestClient

SRC_DIR = pathlib.Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from scorer import MAX_BATCH_SIZE, app

client = TestClient(app)


def test_health_and_readiness_identify_demo_mode():
    health = client.get("/health")
    ready = client.get("/ready")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "mode": "demo",
        "model": "demo-simulator",
    }
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "mode": "demo",
        "model": "demo-simulator",
    }


def test_score_flow_labels_simulated_model():
    response = client.post(
        "/api/v1/score",
        json={
            "amount": 5000,
            "currency": "USD",
            "country": "US",
            "category": "retail",
            "description": "Example purchase",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "demo-simulator"
    assert payload["llm_skipped"] is False
    assert 0 <= payload["risk_score"] <= 100


def test_invalid_transaction_inputs_are_rejected():
    negative = client.post("/api/v1/score", json={"amount": -1})
    invalid_country = client.post(
        "/api/v1/score",
        json={"amount": 50, "country": "NOT-A-COUNTRY"},
    )

    assert negative.status_code == 422
    assert invalid_country.status_code == 422


def test_batch_requires_one_to_maximum_transactions():
    empty = client.post("/api/v1/batch", json={"transactions": []})
    oversized = client.post(
        "/api/v1/batch",
        json={"transactions": [{"amount": 50}] * (MAX_BATCH_SIZE + 1)},
    )

    assert empty.status_code == 422
    assert oversized.status_code == 422


def test_stats_expose_failures_separately_from_intentional_skips():
    response = client.get("/api/v1/stats")

    assert response.status_code == 200
    assert "llm_failures" in response.json()
