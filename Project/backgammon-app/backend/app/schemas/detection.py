from pydantic import BaseModel
from typing import Dict, List, Optional


class CheckerCount(BaseModel):
    white: int
    black: int


class FrameResult(BaseModel):
    turn: int
    player: str
    dice: List[int]
    cube: int
    points: Dict[str, CheckerCount]
    bar: CheckerCount
    borne_off: CheckerCount


class DetectionResponse(BaseModel):
    video_filename: str
    total_frames_analyzed: int
    fps: float
    duration_sec: float
    frames: List[FrameResult]
    error: Optional[str] = None
