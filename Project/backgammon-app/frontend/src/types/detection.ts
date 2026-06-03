export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Piece {
  id: string
  color: 'white' | 'black'
  point: number          // 1-24, 0=bar, 25=off
  bounding_box: BoundingBox
  confidence: number
}

export interface FrameResult {
  frame_index: number
  timestamp_sec: number
  pieces: Piece[]
  board_bounding_box?: BoundingBox
  raw_image_base64?: string
}

export interface DetectionResponse {
  video_filename: string
  total_frames_analyzed: number
  fps: number
  duration_sec: number
  frames: FrameResult[]
  error?: string
}
