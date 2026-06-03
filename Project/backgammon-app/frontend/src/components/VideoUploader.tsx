import { useRef, useState, DragEvent, ChangeEvent } from 'react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

const ACCEPTED = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska']

export function VideoUploader({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = (file: File) => {
    if (!ACCEPTED.includes(file.type)) {
      alert('Unsupported format. Please upload mp4, avi, mov, or mkv.')
      return
    }
    onFile(file)
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div
      className={`uploader ${dragging ? 'uploader--drag' : ''} ${disabled ? 'uploader--disabled' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        style={{ display: 'none' }}
        onChange={onChange}
        disabled={disabled}
      />
      <div className="uploader__icon">▶</div>
      <p className="uploader__label">Drop a video file here</p>
      <p className="uploader__sub">mp4 · avi · mov · mkv &nbsp;·&nbsp; max 500 MB</p>
    </div>
  )
}
