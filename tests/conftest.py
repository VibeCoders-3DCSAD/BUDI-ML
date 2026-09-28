from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from app.main import app
from app.models.registry import ModelRegistry
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def trusted_history_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODIN_TRUSTED_HISTORY_SHARED_SECRET", "test-history-secret")


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        assert c.get("/ready").json()["status"] == "ready"
        yield c


@pytest.fixture(scope="session")
def registry() -> ModelRegistry:
    return app.state.registry


def load_transactions(n: int = 120, seed: int = 7) -> list[dict]:
    import pandas as pd

    txn = pd.read_parquet(
        "training/synth/transactions.parquet",
        columns=["transaction_id", "date", "amount", "category", "transaction_type"],
    )
    txn = txn.sample(n, random_state=seed)
    return [
        {
            "transaction_id": str(r.transaction_id),
            "date": str(r.date),
            "amount": float(r.amount),
            "category": r.category,
            "transaction_type": r.transaction_type,
        }
        for _, r in txn.iterrows()
    ]


def trusted_post(client: TestClient, path: str, payload: dict, user_id: str = "test-user-1"):
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(
        b"test-history-secret", b":".join([b"odin_api", user_id.encode(), body]), hashlib.sha256
    ).hexdigest()
    return client.post(
        path,
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Odin-User-Id": user_id,
            "X-Odin-History-Source": "odin_api",
            "X-Odin-History-Signature": signature,
        },
    )
