import { useEffect } from 'react'
import './Snackbar.css'

interface SnackbarProps {
  message: string
  type?: 'success' | 'error' | 'info' | 'warning'
  isOpen: boolean
  onClose: () => void
  duration?: number
}

function Snackbar({
  message,
  type = 'info',
  isOpen,
  onClose,
  duration = 4000
}: SnackbarProps) {
  useEffect(() => {
    if (isOpen) {
      const timer = setTimeout(() => {
        onClose()
      }, duration)

      return () => clearTimeout(timer)
    }
  }, [isOpen, duration, onClose])

  if (!isOpen) return null

  return (
    <div className={`snackbar snackbar-${type} ${isOpen ? 'snackbar-show' : ''}`}>
      <div className="snackbar-content">
        <span className="snackbar-message">{message}</span>
        <button className="snackbar-close" onClick={onClose}>
          ×
        </button>
      </div>
    </div>
  )
}

export default Snackbar

