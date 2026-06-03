import type { DetectionResponse, FrameResult } from '../types/detection'
import { useState } from 'react'

interface Props {
  result: DetectionResponse
}

export function DetectionResult({ result }: Props) {
  const [selectedFrame, setSelectedFrame] = useState<FrameResult | null>(
    result.frames[0] ?? null
  )

  const whitePieces = selectedFrame?.pieces.filter(p => p.color === 'white') ?? []
  const blackPieces = selectedFrame?.pieces.filter(p => p.color === 'black') ?? []

  return (
    <div className="result">
      <div className="result__meta">
        <div className="result__stat">
          <span className="result__stat-value">{result.total_frames_analyzed}</span>
          <span className="result__stat-label">Frames</span>
        </div>
        <div className="result__stat">
          <span className="result__stat-value">{result.fps.toFixed(1)}</span>
          <span className="result__stat-label">FPS</span>
        </div>
        <div className="result__stat">
          <span className="result__stat-value">{result.duration_sec.toFixed(1)}s</span>
          <span className="result__stat-label">Duration</span>
        </div>
      </div>

      {result.error && (
        <div className="result__error">{result.error}</div>
      )}

      {result.frames.length > 0 && (
        <>
          <div className="result__frames">
            {result.frames.map(f => (
              <button
                key={f.frame_index}
                className={`result__frame-btn ${selectedFrame?.frame_index === f.frame_index ? 'result__frame-btn--active' : ''}`}
                onClick={() => setSelectedFrame(f)}
              >
                {f.timestamp_sec.toFixed(2)}s
              </button>
            ))}
          </div>

          {selectedFrame && (
            <div className="result__frame-detail">
              {selectedFrame.raw_image_base64 && (
                <img
                  className="result__frame-img"
                  src={`data:image/jpeg;base64,${selectedFrame.raw_image_base64}`}
                  alt={`Frame at ${selectedFrame.timestamp_sec.toFixed(2)}s`}
                />
              )}
              <div className="result__pieces">
                <div className="result__piece-group">
                  <h4>White ({whitePieces.length})</h4>
                  {whitePieces.map(p => (
                    <div key={p.id} className="result__piece result__piece--white">
                      Point {p.point} &nbsp; <span>{(p.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
                <div className="result__piece-group">
                  <h4>Black ({blackPieces.length})</h4>
                  {blackPieces.map(p => (
                    <div key={p.id} className="result__piece result__piece--black">
                      Point {p.point} &nbsp; <span>{(p.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
