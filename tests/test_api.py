"""API tests. Skipped automatically if fastapi / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app, get_pipeline  # noqa: E402


@pytest.fixture
def client(settings, monkeypatch):
    get_pipeline.cache_clear()
    yield TestClient(app)
    get_pipeline.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_query_delete_flow(client):
    r = client.post("/v1/documents", files={"file": ("policy.txt", b"Employees get 24 days of annual leave.")})
    assert r.status_code == 201 and r.json()["chunks"] == 1
    doc_id = r.json()["doc_id"]

    q = client.post("/v1/query", json={"question": "How many days of annual leave?"}).json()
    assert q["grounded"] and "24" in q["answer"] and q["sources"][0]["source"] == "policy.txt"

    assert client.get("/v1/documents").json()[0]["doc_id"] == doc_id
    assert client.delete(f"/v1/documents/{doc_id}").json()["chunks_removed"] == 1
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 404


def test_bad_uploads(client):
    assert client.post("/v1/documents", files={"file": ("x.exe", b"abc")}).status_code == 415
    assert client.post("/v1/documents", files={"file": ("x.txt", b"   ")}).status_code == 422


def test_search_diagnostics(client):
    client.post("/v1/documents", files={"file": ("person.txt", b"The person's name is Ada Lovelace.")})
    r = client.post("/v1/search", json={"question": "give me name of the person mentioned"})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"confidence", "min_score", "fallback_min_score", "best_scores", "hits"}
    assert set(body["best_scores"]) == {"dense", "lexical"}


def test_query_validation(client):
    assert client.post("/v1/query", json={"question": "a"}).status_code == 422


def test_api_key_enforced(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    get_settings.cache_clear()
    assert client.post("/v1/query", json={"question": "anything here"}).status_code == 401
    ok = client.post("/v1/query", json={"question": "anything here"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200        # health stays open for probes
