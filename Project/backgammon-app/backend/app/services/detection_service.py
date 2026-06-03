import os
import uuid
import shutil
from pathlib import Path
from app.core.config import settings
from app.schemas.detection import DetectionResponse

# ── Replace this import with your actual detection module ──────────────────────
# from detection_module import BackgammonDetector
# detector = BackgammonDetector()
# ──────────────────────────────────────────────────────────────────────────────


class DetectionService:
    """
    Wraps your BackgammonDetector module.

    Replace the `_run_detection` method body with your actual module calls.
    The rest of the service (file handling, cleanup) is ready to use as-is.
    """

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
        """
        TODO: Replace this stub with your actual detection module call.

        Example:
            result = detector.process_video(str(video_path))
            return DetectionResponse(
                video_filename=filename,
                total_frames_analyzed=result.frame_count,
                fps=result.fps,
                duration_sec=result.duration,
                frames=[...],
            )
        """
        # ── STUB — remove and replace with real detection ─────────────────────
        return DetectionResponse(
            video_filename=filename,
            total_frames_analyzed=0,
            fps=0.0,
            duration_sec=0.0,
            frames=[],
            error="Detection module not yet connected. Replace stub in detection_service.py.",
        )
        # ──────────────────────────────────────────────────────────────────────


detection_service = DetectionService()
