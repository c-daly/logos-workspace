"""Eval tests for hermes #100 — /translate_visual endpoint.

Black-box HTTP tests against a running hermes instance.
The endpoint accepts raw media, runs V-JEPA + MLPTranslator,
and returns CLIP-space embeddings (768-dim).
"""
import math
import os
import struct
import zlib

import httpx
import pytest

HERMES_URL = os.environ.get("HERMES_URL", "http://localhost:17000")
TIMEOUT = 30.0


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
# Endpoint contract tests
# ---------------------------------------------------------------------------

class TestTranslateVisualEndpoint:
    """POST /translate_visual contract tests."""

    def test_returns_embedding(self, client, tiny_png):
        """Basic smoke: upload image, get CLIP embedding back."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding" in body
        assert "dim" in body
        assert "model" in body

    def test_embedding_shape(self, client, tiny_png):
        """Embedding must be exactly 768-dimensional."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        body = resp.json()
        assert body["dim"] == 768
        assert len(body["embedding"]) == 768
        print(f"[METRIC] embedding_dim={body['dim']}")

    def test_values_finite(self, client, tiny_png):
        """All embedding values must be finite (no NaN/Inf)."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        embedding = resp.json()["embedding"]
        assert all(math.isfinite(v) for v in embedding), "Embedding contains NaN or Inf"
        print("[METRIC] values_finite=1.0")

    def test_model_identifier(self, client, tiny_png):
        """Response must identify the model."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        assert resp.json()["model"] == "jepa-v7"

    def test_embedding_normalized(self, client, tiny_png):
        """Output should be L2-normalized (magnitude ~1.0)."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        embedding = resp.json()["embedding"]
        magnitude = math.sqrt(sum(v * v for v in embedding))
        assert abs(magnitude - 1.0) < 0.01, f"Expected L2-norm ~1.0, got {magnitude}"


class TestTranslateVisualErrorHandling:
    """Error handling and edge cases."""

    def test_no_file_returns_422(self, client):
        """Missing file upload should return 422."""
        resp = client.post("/translate_visual")
        assert resp.status_code == 422

    def test_invalid_media_returns_error(self, client):
        """Garbage input should return an error, never silent bad data."""
        resp = client.post(
            "/translate_visual",
            files={"file": ("garbage.bin", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert resp.status_code in (400, 500)
        body = resp.json()
        assert "detail" in body or "error" in body, "Error response must include a message"


class TestTranslateVisualHealth:
    """Health check integration."""

    def test_health_check_passes(self, client):
        """Health endpoint should report healthy when providers are loaded."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_after_translate(self, client, tiny_png):
        """Health should still pass after a successful translation call."""
        client.post(
            "/translate_visual",
            files={"file": ("test.png", tiny_png, "image/png")},
        )
        resp = client.get("/health")
        assert resp.status_code == 200


class TestTranslateVisualStability:
    """Stability under repeated calls."""

    def test_repeated_calls_consistent_shape(self, client, tiny_png):
        """Multiple calls should all return 768-dim embeddings."""
        for _ in range(5):
            resp = client.post(
                "/translate_visual",
                files={"file": ("test.png", tiny_png, "image/png")},
            )
            assert resp.status_code == 200
            assert len(resp.json()["embedding"]) == 768

    def test_deterministic_for_same_input(self, client, tiny_png):
        """Same input should produce same embedding."""
        results = []
        for _ in range(3):
            resp = client.post(
                "/translate_visual",
                files={"file": ("test.png", tiny_png, "image/png")},
            )
            results.append(resp.json()["embedding"])
        for r in results[1:]:
            assert r == results[0], "Same input produced different embeddings"
