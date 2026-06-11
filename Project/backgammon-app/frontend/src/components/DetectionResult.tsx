import { useEffect, useMemo, useState } from 'react'
import type { DetectionResponse, FrameResult } from '../types/detection'

interface Props {
  result: DetectionResponse
}

export function DetectionResult({ result }: Props) {
  const [selectedTurn, setSelectedTurn] = useState<number | null>(
    result.frames[0]?.turn ?? null
  )

  useEffect(() => {
    setSelectedTurn(result.frames[0]?.turn ?? null)
  }, [result])

  const selectedFrame = useMemo<FrameResult | null>(
    () => result.frames.find(frame => frame.turn === selectedTurn) ?? result.frames[0] ?? null,
    [result.frames, selectedTurn]
  )

  const pointEntries = useMemo(
    () => Object.entries(selectedFrame?.points ?? {}).sort((a, b) => Number(a[0]) - Number(b[0])),
    [selectedFrame]
  )

  const formatCount = (white: number, black: number) => `${white} / ${black}`

  const formatDice = (dice: number[]) => (dice.length > 0 ? dice.join(' · ') : '—')

  const renderCount = (label: string, white: number, black: number) => (
    <div className="result__count-card">
      <span className="result__count-label">{label}</span>
      <span className="result__count-value">{formatCount(white, black)}</span>
      <span className="result__count-sub">White / Black</span>
    </div>
  )

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

      <div className="result__file">{result.video_filename}</div>

      {result.error && (
        <div className="result__error">{result.error}</div>
      )}

      {result.frames.length > 0 && (
        <>
          <div className="result__frames">
            {result.frames.map((f, idx) => (
              <button
                key={idx}
                className={`result__frame-btn ${selectedFrame?.turn === f.turn ? 'result__frame-btn--active' : ''}`}
                onClick={() => setSelectedTurn(f.turn)}
              >
                Turn {f.turn}
                <span className="result__frame-btn-meta">
                  {f.player} · D {formatDice(f.dice)} · C {f.cube}
                </span>
              </button>
            ))}
          </div>

          {selectedFrame && (
            <div className="result__frame-detail">
              <div className="result__summary-grid">
                <div className="result__summary-card">
                  <span className="result__summary-label">Player</span>
                  <span className="result__summary-value">{selectedFrame.player}</span>
                </div>
                <div className="result__summary-card">
                  <span className="result__summary-label">Dice</span>
                  <span className="result__summary-value">{formatDice(selectedFrame.dice)}</span>
                </div>
                <div className="result__summary-card">
                  <span className="result__summary-label">Cube</span>
                  <span className="result__summary-value">{selectedFrame.cube}</span>
                </div>
              </div>

              <div className="result__counters">
                {renderCount('Bar', selectedFrame.bar.white, selectedFrame.bar.black)}
                {renderCount('Borne off', selectedFrame.borne_off.white, selectedFrame.borne_off.black)}
              </div>

              <div className="result__points">
                <h4 className="result__section-title">Points</h4>
                <div className="result__point-list">
                  {pointEntries.length > 0 ? pointEntries.map(([point, count]) => (
                    <div key={point} className="result__point-row">
                      <span className="result__point-name">Point {point}</span>
                      <span className="result__point-value">{formatCount(count.white, count.black)}</span>
                    </div>
                  )) : (
                    <div className="result__empty">No point data available.</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
