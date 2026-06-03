from pydantic import BaseModel
from typing import List, Optional


class Point(BaseModel):
    x: float
    y: float


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Piece(BaseModel):
    id: str
    color: str           # "white" | "black"
    point: int           # backgammon point (1-24), 0=bar, 25=off
    bounding_box: BoundingBox
    confidence: float


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    pieces: List[Piece]
    board_bounding_box: Optional[BoundingBox] = None
    raw_image_base64: Optional[str] = None  # optional: annotated frame


class DetectionResponse(BaseModel):
    video_filename: str
    total_frames_analyzed: int
    fps: float
    duration_sec: float
    frames: List[FrameResult]
    error: Optional[str] = None
