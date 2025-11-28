import { useState, useRef, useEffect } from 'react'
import './VolumeSlider.css'

interface VolumeSliderProps {
  label?: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  showValue?: boolean
  disabled?: boolean
  className?: string
}

function VolumeSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  showValue = true,
  disabled = false,
  className = ''
}: VolumeSliderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const sliderRef = useRef<HTMLDivElement>(null)

  const percentage = ((value - min) / (max - min)) * 100

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled) return
    setIsDragging(true)
    updateValueFromEvent(e)
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging || disabled) return
    updateValueFromEvent(e)
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  const handleTouchStart = (e: React.TouchEvent) => {
    if (disabled) return
    setIsDragging(true)
    updateValueFromTouch(e)
  }

  const handleTouchMove = (e: TouchEvent) => {
    if (!isDragging || disabled) return
    updateValueFromTouch(e)
  }

  const handleTouchEnd = () => {
    setIsDragging(false)
  }

  const updateValueFromEvent = (e: MouseEvent | React.MouseEvent) => {
    if (!sliderRef.current) return

    const rect = sliderRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    const newValue = min + (percentage / 100) * (max - min)
    const steppedValue = Math.round(newValue / step) * step
    const clampedValue = Math.max(min, Math.min(max, steppedValue))

    onChange(clampedValue)
  }

  const updateValueFromTouch = (e: TouchEvent | React.TouchEvent) => {
    if (!sliderRef.current) return

    let clientX: number
    if ('touches' in e && e.touches.length > 0) {
      clientX = e.touches[0].clientX
    } else if ('changedTouches' in e && e.changedTouches.length > 0) {
      clientX = e.changedTouches[0].clientX
    } else {
      return
    }
    
    const rect = sliderRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    const newValue = min + (percentage / 100) * (max - min)
    const steppedValue = Math.round(newValue / step) * step
    const clampedValue = Math.max(min, Math.min(max, steppedValue))

    onChange(clampedValue)
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return
    const newValue = parseFloat(e.target.value)
    onChange(newValue)
  }

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.addEventListener('touchmove', handleTouchMove)
      document.addEventListener('touchend', handleTouchEnd)

      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
        document.removeEventListener('touchmove', handleTouchMove)
        document.removeEventListener('touchend', handleTouchEnd)
      }
    }
  }, [isDragging])

  return (
    <div className={`volume-slider-group ${className}`}>
      {label && (
        <label className="volume-slider-label">
          {label}
          {showValue && (
            <span className="volume-slider-value">
              {step < 1 ? value.toFixed(step.toString().split('.')[1]?.length || 1) : Math.round(value)}
            </span>
          )}
        </label>
      )}
      <div
        ref={sliderRef}
        className={`volume-slider-container ${disabled ? 'disabled' : ''}`}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
      >
        <div className="volume-slider-track">
          <div
            className="volume-slider-fill"
            style={{ width: `${percentage}%` }}
          />
          <div
            className="volume-slider-thumb"
            style={{ left: `${percentage}%` }}
          />
        </div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={handleInputChange}
          className="volume-slider-input"
          disabled={disabled}
          aria-label={label || 'Volume slider'}
        />
      </div>
      {!label && showValue && (
        <span className="volume-slider-value">
          {step < 1 ? value.toFixed(step.toString().split('.')[1]?.length || 1) : Math.round(value)}
        </span>
      )}
    </div>
  )
}

export default VolumeSlider

