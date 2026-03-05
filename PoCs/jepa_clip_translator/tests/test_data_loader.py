"""Tests for MSR-VTT data loader."""

import json
import pathlib

import numpy as np
import pytest

from data_loader import MSRVTTLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_annotations(path: pathlib.Path, n_videos: int = 5, captions_per: int = 3):
    """Write a minimal MSR-VTT-style annotation JSON to *path*."""
    videos = []
    sentences = []
    for i in range(n_videos):
        vid = f"video{i}"
        videos.append({
            "video_id": vid,
            "url": f"https://example.com/{vid}.mp4",
            "category": 0,
            "start time": 0.0,
            "end time": 10.0,
        })
        for j in range(captions_per):
            sentences.append({
                "video_id": vid,
                "sen_id": i * captions_per + j,
                "caption": f"Caption {j} for {vid}.",
            })
    ann = {"videos": videos, "sentences": sentences}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ann))
    return ann


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMSRVTTLoaderInit:
    """Constructor sanity checks."""

    def test_loader_init(self, tmp_path):
        loader = MSRVTTLoader(data_dir=str(tmp_path), subset_size=10)
        assert loader.data_dir == pathlib.Path(tmp_path)
        assert loader.subset_size == 10

    def test_loader_defaults(self, tmp_path):
        loader = MSRVTTLoader(data_dir=str(tmp_path))
        assert loader.subset_size == 50  # default from config


class TestParseAnnotations:
    """Annotation parsing produces the expected structures."""

    def test_load_annotations_structure(self, tmp_path):
        n_videos, captions_per = 5, 3
        ann_path = tmp_path / "annotations.json"
        _make_mock_annotations(ann_path, n_videos=n_videos, captions_per=captions_per)

        loader = MSRVTTLoader(data_dir=str(tmp_path))
        videos, captions = loader.parse_annotations(str(ann_path))

        # videos list length matches
        assert len(videos) == n_videos
        # each video entry has a video_id key
        assert all("video_id" in v for v in videos)
        # captions dict keys match video ids
        assert set(captions.keys()) == {f"video{i}" for i in range(n_videos)}
        # each video has the right number of captions
        for vid_id, caps in captions.items():
            assert len(caps) == captions_per
            assert all(isinstance(c, str) for c in caps)

    def test_empty_annotations(self, tmp_path):
        """An annotation file with no videos/sentences returns empty structures."""
        ann_path = tmp_path / "empty.json"
        ann_path.write_text(json.dumps({"videos": [], "sentences": []}))

        loader = MSRVTTLoader(data_dir=str(tmp_path))
        videos, captions = loader.parse_annotations(str(ann_path))
        assert videos == []
        assert captions == {}

    def test_missing_file_raises(self, tmp_path):
        """Parsing a non-existent file raises FileNotFoundError."""
        loader = MSRVTTLoader(data_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.parse_annotations(str(tmp_path / "nonexistent.json"))


class TestLoadVideoFrames:
    """Frame loading edge cases (no real video files)."""

    def test_missing_video_returns_none(self, tmp_path):
        """When the video file doesn't exist on disk, return None."""
        loader = MSRVTTLoader(data_dir=str(tmp_path))
        result = loader.load_video_frames("video9999")
        assert result is None

    def test_max_frames_stored(self, tmp_path):
        """max_frames parameter is respected (tested via the method signature)."""
        loader = MSRVTTLoader(data_dir=str(tmp_path))
        # Just confirm the method accepts max_frames without error
        result = loader.load_video_frames("nonexistent", max_frames=8)
        assert result is None


class TestHelpers:
    """Coverage for internal helper methods."""

    def test_find_local_videos_empty(self, tmp_path):
        """No videos directory -> empty dict."""
        loader = MSRVTTLoader(data_dir=str(tmp_path))
        found = loader._find_local_videos()
        assert found == {}

    def test_find_local_videos_discovers_mp4(self, tmp_path):
        """Discovers .mp4 files in videos/ subdirectory."""
        vids_dir = tmp_path / "videos"
        vids_dir.mkdir()
        (vids_dir / "video0.mp4").write_bytes(b"\x00")
        (vids_dir / "video1.mp4").write_bytes(b"\x00")
        (vids_dir / "readme.txt").write_bytes(b"\x00")  # non-video, ignored

        loader = MSRVTTLoader(data_dir=str(tmp_path))
        found = loader._find_local_videos()
        assert "video0" in found
        assert "video1" in found
        assert "readme" not in found
