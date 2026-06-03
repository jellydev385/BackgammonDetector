import { useState, useCallback } from 'react'
import { detectVideo } from '../api/detectionApi'
import type { DetectionResponse } from '../types/detection'

type Status = 'idle' | 'uploading' | 'processing' | 'done' | 'error'

export function useDetection() {
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<DetectionResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const run = useCallback(async (file: File) => {
    setStatus('uploading')
    setError(null)
    setResult(null)
    setProgress(0)

    // Simulate upload progress (real progress requires XHR)
    const ticker = setInterval(() => {
      setProgress(p => Math.min(p + 10, 80))
    }, 200)

    try {
      setStatus('processing')
      const data = await detectVideo(file)
      clearInterval(ticker)
      setProgress(100)
      setResult(data)
      setStatus('done')
    } catch (err) {
      clearInterval(ticker)
      setError(err instanceof Error ? err.message : 'Detection failed')
      setStatus('error')
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setResult(null)
    setError(null)
    setProgress(0)
  }, [])

  return { status, result, error, progress, run, reset }
}
