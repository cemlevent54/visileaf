import './Input.css'

interface InputProps {
  label: string
  type?: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  placeholder?: string
  required?: boolean
  error?: string
  min?: number
  max?: number
  step?: number
}

function Input({
  label,
  type = 'text',
  value,
  onChange,
  placeholder = '',
  required = false,
  error,
  min,
  max,
  step
}: InputProps) {
  return (
    <div className="input-group">
      <label className="input-label" htmlFor={label.toLowerCase().replace(/\s+/g, '-')}>
        {label}
        {required && <span className="required">*</span>}
      </label>
      <input
        id={label.toLowerCase().replace(/\s+/g, '-')}
        type={type}
        className={`input-field ${error ? 'input-error' : ''}`}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        required={required}
        min={min}
        max={max}
        step={step}
      />
      {error && <span className="input-error-message">{error}</span>}
    </div>
  )
}

export default Input

