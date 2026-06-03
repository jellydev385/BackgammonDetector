# Backgammon Detection App

A web app for detecting backgammon board state from video files.

- **Backend**: Python + FastAPI
- **Frontend**: TypeScript + React + Vite

---

## Project Structure

```
backgammon-app/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point
│   │   ├── api/routes/
│   │   │   ├── detection.py         # POST /api/detect
│   │   │   └── health.py            # GET  /api/health
│   │   ├── core/config.py           # Settings (.env)
│   │   ├── services/
│   │   │   └── detection_service.py # ← Connect your module here
│   │   └── schemas/detection.py     # Pydantic request/response types
│   ├── detection_module/            # ← Place your detection module here
│   └── requirements.txt
│
└── frontend/
    └── src/
        ├── api/detectionApi.ts      # Typed fetch client
        ├── components/
        │   ├── VideoUploader.tsx    # Drag-and-drop uploader
        │   └── DetectionResult.tsx  # Results display
        ├── hooks/useDetection.ts    # Detection state hook
        └── types/detection.ts      # TypeScript types
```

---

## Quick Start

### 1. Backend

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env config
cp .env.example .env

# Start the server
uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

App available at: http://localhost:5173

---

## Connecting Your Detection Module

1. Copy your detection module into `backend/detection_module/`
2. Open `backend/app/services/detection_service.py`
3. Replace the stub in `_run_detection()` with your actual module call:

```python
from detection_module import BackgammonDetector
detector = BackgammonDetector()

def _run_detection(self, video_path: Path, filename: str) -> DetectionResponse:
    result = detector.process_video(str(video_path))
    return DetectionResponse(
        video_filename=filename,
        total_frames_analyzed=result.frame_count,
        fps=result.fps,
        duration_sec=result.duration,
        frames=[...],  # map your results to FrameResult
    )
```

---

## API Reference

### `POST /api/detect`

Upload a video file for detection.

**Request**: `multipart/form-data` with `file` field (mp4, avi, mov, mkv — max 500 MB)

**Response**:
```json
{
  "video_filename": "game.mp4",
  "total_frames_analyzed": 120,
  "fps": 30.0,
  "duration_sec": 4.0,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_sec": 0.0,
      "pieces": [
        {
          "id": "piece_0",
          "color": "white",
          "point": 6,
          "bounding_box": { "x": 120, "y": 80, "width": 40, "height": 40 },
          "confidence": 0.97
        }
      ]
    }
  ]
}
```
