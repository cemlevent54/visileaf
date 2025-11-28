import './RadioButton.css'

interface RadioOption {
  value: string
  label: string
  disabled?: boolean
}

interface RadioButtonProps {
  label?: string
  name: string
  options: RadioOption[]
  value: string
  onChange: (value: string) => void
  required?: boolean
  error?: string
  inline?: boolean
  className?: string
}

function RadioButton({
  label,
  name,
  options,
  value,
  onChange,
  required = false,
  error,
  inline = false,
  className = ''
}: RadioButtonProps) {
  const handleChange = (optionValue: string) => {
    onChange(optionValue)
  }

  const inputId = (option: RadioOption) => {
    return `${name}-${option.value}`.toLowerCase().replace(/\s+/g, '-')
  }

  return (
    <div className={`radio-group ${inline ? 'radio-inline' : ''} ${className}`}>
      {label && (
        <label className="radio-label">
          {label}
          {required && <span className="required">*</span>}
        </label>
      )}
      <div className={`radio-options ${inline ? 'radio-options-inline' : ''}`}>
        {options.map((option) => {
          const id = inputId(option)
          const isChecked = value === option.value

          return (
            <div
              key={option.value}
              className={`radio-option ${option.disabled ? 'disabled' : ''}`}
            >
              <input
                type="radio"
                id={id}
                name={name}
                value={option.value}
                checked={isChecked}
                onChange={() => handleChange(option.value)}
                disabled={option.disabled}
                className="radio-input"
                required={required}
              />
              <label
                htmlFor={id}
                className={`radio-label-option ${isChecked ? 'checked' : ''}`}
              >
                <span className="radio-custom">
                  <span className="radio-inner" />
                </span>
                <span className="radio-text">{option.label}</span>
              </label>
            </div>
          )
        })}
      </div>
      {error && <span className="radio-error-message">{error}</span>}
    </div>
  )
}

export default RadioButton

