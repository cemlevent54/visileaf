import './Checkbox.css'

interface CheckboxOption {
  value: string
  label: string
  disabled?: boolean
}

interface CheckboxProps {
  label?: string
  name: string
  options?: CheckboxOption[]
  value?: string | string[] | boolean
  onChange: (value: string | string[]) => void
  required?: boolean
  error?: string
  inline?: boolean
  single?: boolean
  className?: string
}

function Checkbox({
  label,
  name,
  options,
  value,
  onChange,
  required = false,
  error,
  inline = false,
  single = false,
  className = ''
}: CheckboxProps) {
  // Single checkbox (controlled by boolean-like value)
  if (single && !options) {
    const isChecked = 
      value === 'checked' || 
      value === true || 
      value === 'true' ||
      (Array.isArray(value) && value.length > 0) ||
      (typeof value === 'string' && value !== '')

    const handleSingleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      onChange(e.target.checked ? 'checked' : '')
    }

    const inputId = name.toLowerCase().replace(/\s+/g, '-')

    return (
      <div className={`checkbox-group checkbox-single ${className}`}>
        <div className="checkbox-option">
          <input
            type="checkbox"
            id={inputId}
            name={name}
            checked={isChecked}
            onChange={handleSingleChange}
            className="checkbox-input"
            required={required}
          />
          <label htmlFor={inputId} className="checkbox-label-option">
            <span className="checkbox-custom">
              <span className="checkbox-checkmark">✓</span>
            </span>
            {label && <span className="checkbox-text">{label}</span>}
          </label>
        </div>
        {error && <span className="checkbox-error-message">{error}</span>}
      </div>
    )
  }

  // Multiple checkboxes
  if (!options || options.length === 0) {
    return null
  }

  const selectedValues: string[] = Array.isArray(value) 
    ? value.filter((v): v is string => typeof v === 'string')
    : (typeof value === 'string' && value !== '' ? [value] : [])

  const handleChange = (optionValue: string, checked: boolean) => {
    if (checked) {
      const newValues = [...selectedValues, optionValue]
      onChange(newValues)
    } else {
      const newValues = selectedValues.filter((v) => v !== optionValue)
      onChange(newValues)
    }
  }

  const inputId = (option: CheckboxOption) => {
    return `${name}-${option.value}`.toLowerCase().replace(/\s+/g, '-')
  }

  return (
    <div className={`checkbox-group ${inline ? 'checkbox-inline' : ''} ${className}`}>
      {label && (
        <label className="checkbox-label">
          {label}
          {required && <span className="required">*</span>}
        </label>
      )}
      <div className={`checkbox-options ${inline ? 'checkbox-options-inline' : ''}`}>
        {options.map((option) => {
          const id = inputId(option)
          const isChecked = selectedValues.includes(option.value)

          return (
            <div
              key={option.value}
              className={`checkbox-option ${option.disabled ? 'disabled' : ''}`}
            >
              <input
                type="checkbox"
                id={id}
                name={name}
                value={option.value}
                checked={isChecked}
                onChange={(e) => handleChange(option.value, e.target.checked)}
                disabled={option.disabled}
                className="checkbox-input"
              />
              <label
                htmlFor={id}
                className={`checkbox-label-option ${isChecked ? 'checked' : ''}`}
              >
                <span className="checkbox-custom">
                  <span className="checkbox-checkmark">✓</span>
                </span>
                <span className="checkbox-text">{option.label}</span>
              </label>
            </div>
          )
        })}
      </div>
      {error && <span className="checkbox-error-message">{error}</span>}
    </div>
  )
}

export default Checkbox

