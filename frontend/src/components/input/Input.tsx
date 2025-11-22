import './Input.css'

interface InputProps {
  label: string
  type?: string
  value: string
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  placeholder?: string
  required?: boolean
  error?: string
}

function Input({
  label,
  type = 'text',
  value,
  onChange,
  placeholder = '',
  required = false,
  error
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
      />
      {error && <span className="input-error-message">{error}</span>}
    </div>
  )
}

export default Input

