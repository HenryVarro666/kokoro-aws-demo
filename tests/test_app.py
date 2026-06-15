import os

os.environ["SKIP_MODEL_LOAD"] = "1"  # CI must not load the model

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_synthesize_rejects_empty_text():
    with TestClient(app) as client:
        assert client.post("/synthesize", json={"text": ""}).status_code == 422


def test_synthesize_rejects_bad_lang():
    with TestClient(app) as client:
        resp = client.post("/synthesize", json={"text": "hi", "lang": "x"})
        assert resp.status_code == 422


def test_avatar_503_when_pipeline_not_deployed():
    # AVATAR_* env vars are unset in tests, so the endpoint must degrade cleanly.
    with TestClient(app) as client:
        assert client.post("/avatar", json={"text": "hi"}).status_code == 503
        assert client.get("/avatar/abc123").status_code == 503


def test_transcribe_requires_file():
    # Contract check only — missing upload is rejected by FastAPI validation
    # before the ASR model is ever loaded (no model download in CI).
    with TestClient(app) as client:
        assert client.post("/transcribe").status_code == 422


def test_transcribe_is_sync():
    # Regression guard: transcribe MUST stay a sync def. faster-whisper is
    # CPU-bound/blocking; an async def would starve the event loop and make
    # /health hang during a long transcription (ALB then kills the task).
    import inspect

    from app.main import transcribe
    assert not inspect.iscoroutinefunction(transcribe)
