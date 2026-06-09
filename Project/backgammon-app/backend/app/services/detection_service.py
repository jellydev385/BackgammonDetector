import os
import sys
import uuid
import json
from pathlib import Path

import cv2

from app.core.config import settings
from app.schemas.detection import DetectionResponse

# ── Add detection_module to Python path ───────────────────────────────────────
# detection_service.py -> backend/app/services/
# backend root is parents[2], so detection_module lives at backend/detection_module
_backend_root = Path(__file__).resolve().parents[2]
_detection_module_path = _backend_root / "detection_module"
if not _detection_module_path.exists():
    raise RuntimeError(f"detection_module not found at {_detection_module_path}")
if str(_detection_module_path) not in sys.path:
    sys.path.insert(0, str(_detection_module_path))

from BackgammonCV import BackgammonCV  # type: ignore[import-not-found]
# ──────────────────────────────────────────────────────────────────────────────


class DetectionService:
    def __init__(self):
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    def save_upload(self, file_bytes: bytes, original_filename: str) -> Path:
        ext = Path(original_filename).suffix
        tmp_path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4()}{ext}"
        tmp_path.write_bytes(file_bytes)
        return tmp_path

    def run(self, file_bytes: bytes, filename: str) -> DetectionResponse:
        tmp_path = self.save_upload(file_bytes, filename)
        try:
            return self._run_detection(tmp_path, filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_detection(self, video_path: Path, filename: str) -> DetectionResponse:

        bCV = BackgammonCV(enable_ui=False)

        # ── 1. Open video ──────────────────────────────────────────────────────
        bCV.video = cv2.VideoCapture(str(video_path))
        if not bCV.video.isOpened():
            return DetectionResponse(
                video_filename=filename,
                total_frames_analyzed=0,
                fps=0.0,
                duration_sec=0.0,
                frames=[],
                error="Failed to open video file.",
            )

        # ── 2. Set video stats (mirrors Main.py) ───────────────────────────────
        bCV.total_frames = int(bCV.video.get(cv2.CAP_PROP_FRAME_COUNT))
        bCV.fps = float(bCV.video.get(cv2.CAP_PROP_FPS))
        bCV.duration = bCV.total_frames / bCV.fps if bCV.fps > 0 else 0.0
        bCV.snapshots.snapshots_per_second = int(bCV.fps)
        bCV.snapshots.total_snapshots = int(bCV.total_frames / bCV.fps) if bCV.fps > 0 else 0

        # ── 3. Read first frame ────────────────────────────────────────────────
        bCV.video.set(1, 0)
        ret, bCV.frame = bCV.video.read()
        if not ret:
            bCV.close()
            return DetectionResponse(
                video_filename=filename,
                total_frames_analyzed=0,
                fps=bCV.fps,
                duration_sec=bCV.duration,
                frames=[],
                error="Failed to read first frame from video.",
            )

        # ── 4. Align template (board detection) ────────────────────────────────
        aligned = bCV.alignTemplate(bCV.frame)
        if not aligned or not bCV.template_aligned:
            bCV.close()
            return DetectionResponse(
                video_filename=filename,
                total_frames_analyzed=0,
                fps=bCV.fps,
                duration_sec=bCV.duration,
                frames=[],
                error="Board not detected in video. Check camera angle or lighting.",
            )

        # ── 5. Run detection over all frames ───────────────────────────────────
        movements = bCV.detect()

        # ── 6. Serialize results ───────────────────────────────────────────────
        frames = []
        for movement in movements:
            try:
                if isinstance(movement, dict):
                    frames.append(movement)
                elif hasattr(movement, "exportJSON"):
                    exported = movement.exportJSON(as_string=False)
                    if isinstance(exported, str):
                        exported = json.loads(exported)
                    frames.append(exported)
                elif isinstance(movement, str):
                    frames.append(json.loads(movement))
                else:
                    frames.append(dict(movement))
            except Exception as e:
                frames.append({"error": str(e), "raw": str(movement)})

        total_analyzed = bCV.frame_index
        bCV.close()

        return DetectionResponse(
            video_filename=filename,
            total_frames_analyzed=total_analyzed,
            fps=bCV.fps,
            duration_sec=bCV.duration,
            frames=frames,
        )


detection_service = DetectionService()
