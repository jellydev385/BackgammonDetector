export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface CheckerCount {
  white: number
  black: number
}

export interface FrameResult {
  turn: number
  player: string
  dice: number[]
  cube: number
  points: Record<string, CheckerCount>
  bar: CheckerCount
  borne_off: CheckerCount
}

export interface DetectionResponse {
  video_filename: string
  total_frames_analyzed: number
  fps: number
  duration_sec: number
  frames: FrameResult[]
  error?: string
}
