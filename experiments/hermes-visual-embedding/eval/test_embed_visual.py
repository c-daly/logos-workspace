"""Eval tests for hermes #101 — /embed_visual endpoint.

Black-box HTTP tests against a running hermes instance.
The endpoint accepts raw media and returns embeddings from all
configured visual providers (V-JEPA 1024-dim, CLIP 768-dim).
"""
import math
import os
import struct
import zlib

import httpx
import pytest

HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")
TIMEOUT = 60.0  # model loading may be slow on first call


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def client():
    """HTTP client pointed at the running hermes instance."""
    with httpx.Client(base_url=HERMES_URL, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="session")
def tiny_png():
    """1x1 red PNG — smallest valid image for smoke testing."""
    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_row = b"\x00\xff\x00\x00"  # filter byte + RGB
    idat = _chunk(b"IDAT", zlib.compress(raw_row))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


# ---------------------------------------------------------------------------
# Endpoint contract
# ---------------------------------------------------------------------------

class TestEmbedVisualEndpoint:
    """POST /embed_visual contract tests."""

    def test_returns_embeddings(self, client, tiny_png):
        """Basic smoke: upload image, get embeddings back."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "embeddings" in body
        assert "media_type" in body

    def test_jepa_embedding_present(self, client, tiny_png):
        """V-JEPA embedding should be in response."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        body = resp.json()
        jepa = body["embeddings"].get("jepa")
        assert jepa is not None, "V-JEPA embedding missing from response"
        assert "embedding" in jepa
        assert "dim" in jepa
        assert "model" in jepa

    def test_jepa_embedding_shape(self, client, tiny_png):
        """V-JEPA embedding must be 1024-dimensional."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        jepa = resp.json()["embeddings"]["jepa"]
        assert jepa["dim"] == 1024
        assert len(jepa["embedding"]) == 1024
        print(f"[METRIC] jepa_embedding_dim={jepa['dim']}")

    def test_clip_embedding_present(self, client, tiny_png):
        """CLIP embedding should be in response."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        body = resp.json()
        clip = body["embeddings"].get("clip")
        assert clip is not None, "CLIP embedding missing from response"
        assert "embedding" in clip
        assert "dim" in clip
        assert "model" in clip

    def test_clip_embedding_shape(self, client, tiny_png):
        """CLIP embedding must be 768-dimensional."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        clip = resp.json()["embeddings"]["clip"]
        assert clip["dim"] == 768
        assert len(clip["embedding"]) == 768
        print(f"[METRIC] clip_embedding_dim={clip['dim']}")

    def test_all_values_finite(self, client, tiny_png):
        """All embedding values must be finite (no NaN/Inf)."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        body = resp.json()
        for provider_name, data in body["embeddings"].items():
            for v in data["embedding"]:
                assert math.isfinite(v), f"{provider_name} embedding contains NaN or Inf"
        print("[METRIC] values_finite=1.0")

    def test_media_type_echoed(self, client, tiny_png):
        """Response should echo the detected media type."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        assert resp.json()["media_type"] == "image/png"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEmbedVisualErrorHandling:
    """Error handling and edge cases."""

    def test_no_file_returns_422(self, client):
        """Missing file upload should return 422."""
        resp = client.post("/embed_visual")
        assert resp.status_code == 422

    def test_invalid_media_returns_error(self, client):
        """Garbage input should return an error, never silent bad data."""
        resp = client.post(
            "/embed_visual",
            files={"file": ("garbage.bin", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert resp.status_code in (400, 500)
        body = resp.json()
        assert "detail" in body or "error" in body


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestEmbedVisualHealth:
    """Health check integration."""

    def test_health_check_passes(self, client):
        """Health endpoint should report healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_after_embed(self, client, tiny_png):
        """Health should still pass after a successful embedding call."""
        client.post(
            "/embed_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

class TestEmbedVisualStability:
    """Stability under repeated calls."""

    def test_repeated_calls_consistent_shape(self, client, tiny_png):
        """Multiple calls should all return correct-dimension embeddings."""
        for _ in range(3):
            resp = client.post(
                "/embed_visual",
                files={"file": ("test.png", tiny_png, "image/png")},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert len(body["embeddings"]["jepa"]["embedding"]) == 1024
            assert len(body["embeddings"]["clip"]["embedding"]) == 768

    def test_deterministic_for_same_input(self, client, tiny_png):
        """Same input should produce same embeddings."""
        results = []
        for _ in range(3):
            resp = client.post(
                "/embed_visual",
                files={"file": ("test.png", tiny_png, "image/png")},
            )
            results.append(resp.json()["embeddings"])
        for r in results[1:]:
            assert r["jepa"]["embedding"] == results[0]["jepa"]["embedding"]
            assert r["clip"]["embedding"] == results[0]["clip"]["embedding"]


# ---------------------------------------------------------------------------
# Text embedding regression
# ---------------------------------------------------------------------------

class TestTextEmbeddingRegression:
    """Existing text embedding must still work after refactor."""

    def test_embed_text_still_works(self, client):
        """POST /embed_text should still return embeddings."""
        resp = client.post(
            "/embed_text",
            json={"text": "hello world"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding" in body
        assert len(body["embedding"]) > 0
        print("[METRIC] text_embedding_unchanged=1.0")
