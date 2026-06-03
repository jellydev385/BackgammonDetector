import { useDetection } from './hooks/useDetection'
import { VideoUploader } from './components/VideoUploader'
import { DetectionResult } from './components/DetectionResult'
import './App.css'

export default function App() {
  const { status, result, error, progress, run, reset } = useDetection()

  const isLoading = status === 'uploading' || status === 'processing'

  return (
    <div className="app">
      <header className="header">
        <div className="header__logo">⬡</div>
        <h1 className="header__title">Backgammon Detection</h1>
        <p className="header__sub">Upload a video to analyse board state</p>
      </header>

      <main className="main">
        {status === 'idle' || status === 'error' ? (
          <>
            <VideoUploader onFile={run} disabled={isLoading} />
            {error && <p className="app__error">{error}</p>}
          </>
        ) : isLoading ? (
          <div className="loader">
            <div className="loader__bar">
              <div className="loader__fill" style={{ width: `${progress}%` }} />
            </div>
            <p className="loader__label">
              {status === 'uploading' ? 'Uploading…' : 'Analysing video…'}
            </p>
          </div>
        ) : result ? (
          <>
            <DetectionResult result={result} />
            <button className="btn-reset" onClick={reset}>
              ← Upload another video
            </button>
          </>
        ) : null}
      </main>
    </div>
  )
}
