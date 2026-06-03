import type { DetectionResponse } from '../types/detection'

const BASE_URL = '/api'

export async function detectVideo(file: File): Promise<DetectionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${BASE_URL}/detect`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail ?? `HTTP ${response.status}`)
  }

  return response.json() as Promise<DetectionResponse>
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
