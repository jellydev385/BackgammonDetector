from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.detection import DetectionResponse
from app.services.detection_service import detection_service
from app.core.config import settings

router = APIRouter()

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska"}


@router.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: mp4, avi, mov, mkv",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_VIDEO_SIZE_MB} MB",
        )

    result = detection_service.run(file_bytes, file.filename or "upload.mp4")
    return result
