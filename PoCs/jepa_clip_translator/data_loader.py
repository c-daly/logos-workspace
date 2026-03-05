"""MSR-VTT data loader with download fallback for JEPA-CLIP translator PoC."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import urllib.request
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/ArrowLuo/CLIP4Clip/"
    "master/data/msrvtt_data/MSRVTT_data.json"
)

# Recognised video extensions when scanning for local files.
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".webm", ".mkv"}


class MSRVTTLoader:
    """Load MSR-VTT videos and annotations for the JEPA-CLIP translator PoC.

    Parameters
    ----------
    data_dir:
        Root directory where annotations and videos are stored/downloaded.
    subset_size:
        Number of videos to use.  Defaults to 50 (mirrors ``DataConfig``).
    """

    def __init__(self, data_dir: str, subset_size: int = 50) -> None:
        self.data_dir = pathlib.Path(data_dir)
        self.subset_size = subset_size
        self._video_map: dict[str, pathlib.Path] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_data(self) -> pathlib.Path:
        """Make sure annotations (and optionally videos) are available.

        Downloads the MSR-VTT annotation JSON if it is not already present.
        For videos, checks for a local ``videos/`` directory first; if absent
        it looks for the ``MSRVTT_VIDEOS_URL`` environment variable and falls
        back to printing user instructions.

        Returns the path to the annotation JSON file.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ann_path = self.data_dir / "MSRVTT_data.json"

        # 1. Annotations --------------------------------------------------
        if not ann_path.exists():
            logger.info("Downloading MSR-VTT annotations …")
            self._download_file(ANNOTATIONS_URL, ann_path)
            logger.info("Annotations saved to %s", ann_path)

        # 2. Videos -------------------------------------------------------
        local_vids = self._find_local_videos()
        if not local_vids:
            videos_url = os.environ.get("MSRVTT_VIDEOS_URL")
            if videos_url:
                logger.info("Downloading videos from MSRVTT_VIDEOS_URL …")
                self._download_videos(videos_url)
            else:
                logger.warning(
                    "No local videos found in %s/videos/.  "
                    "Set the MSRVTT_VIDEOS_URL environment variable or "
                    "manually place .mp4 files in that directory.",
                    self.data_dir,
                )

        return ann_path

    def parse_annotations(
        self, ann_path: str
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        """Parse an MSR-VTT annotation JSON file.

        Parameters
        ----------
        ann_path:
            Path to the JSON file (e.g. ``MSRVTT_data.json``).

        Returns
        -------
        videos:
            The raw ``videos`` list from the JSON.
        captions:
            Mapping ``{video_id: [caption_str, …]}``.

        Raises
        ------
        FileNotFoundError
            If *ann_path* does not exist.
        """
        p = pathlib.Path(ann_path)
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(p) as f:
            data = json.load(f)

        videos: list[dict[str, Any]] = data.get("videos", [])

        captions: dict[str, list[str]] = {}
        for sent in data.get("sentences", []):
            vid_id = sent["video_id"]
            captions.setdefault(vid_id, []).append(sent["caption"])

        return videos, captions

    def load_video_frames(
        self, video_id: str, max_frames: int = 64
    ) -> list[np.ndarray] | None:
        """Load uniformly-sampled RGB frames from a video.

        Parameters
        ----------
        video_id:
            MSR-VTT video identifier (e.g. ``"video0"``).
        max_frames:
            Maximum number of frames to return.

        Returns
        -------
        A list of ``(H, W, 3)`` uint8 numpy arrays, or ``None`` if the video
        file is not available locally.
        """
        video_path = self._resolve_video_path(video_id)
        if video_path is None:
            return None

        try:
            import av  # noqa: F811 – imported lazily so tests pass without av
        except ImportError:
            logger.error(
                "PyAV (av) is required to decode video frames.  "
                "Install it with: pip install av"
            )
            return None

        frames: list[np.ndarray] = []
        try:
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            total_frames = stream.frames or 0

            # Determine which frame indices to sample.
            if total_frames > 0 and total_frames > max_frames:
                indices = set(
                    np.linspace(0, total_frames - 1, max_frames, dtype=int)
                )
            else:
                indices = None  # take all (up to max_frames)

            for idx, frame in enumerate(container.decode(video=0)):
                if indices is not None and idx not in indices:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= max_frames:
                    break

            container.close()
        except Exception:
            logger.exception("Failed to decode %s", video_path)
            return None

        return frames if frames else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_local_videos(self) -> dict[str, pathlib.Path]:
        """Scan ``data_dir/videos/`` for video files.

        Returns a dict mapping video stem (e.g. ``"video0"``) to its path.
        """
        if self._video_map is not None:
            return self._video_map

        vids_dir = self.data_dir / "videos"
        found: dict[str, pathlib.Path] = {}
        if vids_dir.is_dir():
            for p in vids_dir.iterdir():
                if p.suffix.lower() in _VIDEO_EXTENSIONS:
                    found[p.stem] = p

        self._video_map = found
        return found

    def _resolve_video_path(self, video_id: str) -> pathlib.Path | None:
        """Return the local path for *video_id*, or ``None``."""
        local = self._find_local_videos()
        return local.get(video_id)

    @staticmethod
    def _download_file(url: str, dest: pathlib.Path) -> None:
        """Download *url* to *dest* (blocking, no progress bar)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))

    def _download_videos(self, url: str) -> None:
        """Download and extract a video archive from *url*.

        The archive is expected to unpack into a ``videos/`` subdirectory
        inside ``data_dir``.
        """
        import shutil
        import tempfile

        vids_dir = self.data_dir / "videos"
        vids_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)

        try:
            logger.info("Downloading videos archive …")
            self._download_file(url, tmp_path)

            logger.info("Extracting to %s …", vids_dir)
            shutil.unpack_archive(str(tmp_path), str(vids_dir))
        finally:
            tmp_path.unlink(missing_ok=True)

        # Invalidate cached video map so next lookup re-scans.
        self._video_map = None
