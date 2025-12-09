import './Select.css'

type Option = {
  value: string
  label: string
}

type SelectProps = {
  label?: string
  value: string
  onChange: (value: string) => void
  options: Option[]
  placeholder?: string
  hint?: string
  disabled?: boolean
}

function Select({
  label,
  value,
  onChange,
  options,
  placeholder = 'Seçin',
  hint,
  disabled = false
}: SelectProps) {
  return (
    <div className="select-wrapper">
      {label && <label className="select-label">{label}</label>}
      <div className="select-control">
        <select
          className="select-input"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">{placeholder}</option>
          {options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <span className="select-arrow">▾</span>
      </div>
      {hint && <p className="select-hint">{hint}</p>}
    </div>
  )
}

export default Select

