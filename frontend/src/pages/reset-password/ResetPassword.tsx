import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams, Link } from 'react-router-dom'
import Navbar from '../../components/navbar/Navbar'
import Input from '../../components/input/Input'
import Snackbar from '../../components/snackbar/Snackbar'
import authService from '../../services/auth.service'
import { useTranslation } from '../../hooks/useTranslation'
import './ResetPassword.css'

function ResetPassword() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { t } = useTranslation()
  const token = searchParams.get('token') || ''

  const [formData, setFormData] = useState({
    code: '',
    password: '',
    confirmPassword: ''
  })

  const [errors, setErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(false)
  const [snackbar, setSnackbar] = useState<{
    isOpen: boolean
    message: string
    type: 'success' | 'error' | 'info' | 'warning'
  }>({
    isOpen: false,
    message: '',
    type: 'info'
  })

  useEffect(() => {
    if (!token) {
      setSnackbar({
        isOpen: true,
        message: t('resetPassword.invalidToken'),
        type: 'error'
      })
      setTimeout(() => {
        navigate('/forgot-password')
      }, 2000)
    }
  }, [token, navigate, t])

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value

    // For code field, only allow 6 digits
    if (field === 'code') {
      const numericValue = value.replace(/\D/g, '').slice(0, 6)
      setFormData(prev => ({
        ...prev,
        [field]: numericValue
      }))
    } else {
      setFormData(prev => ({
        ...prev,
        [field]: value
      }))
    }

    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }))
    }
  }

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {}

    if (!formData.code) {
      newErrors.code = t('resetPassword.codeRequired')
    } else if (formData.code.length !== 6) {
      newErrors.code = t('resetPassword.codeInvalid')
    }

    if (!formData.password) {
      newErrors.password = t('resetPassword.passwordRequired')
    } else if (formData.password.length < 8) {
      newErrors.password = t('resetPassword.passwordMinLength')
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = t('resetPassword.confirmPasswordRequired')
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = t('resetPassword.passwordsDoNotMatch')
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    if (!token) {
      setSnackbar({
        isOpen: true,
        message: t('resetPassword.invalidToken'),
        type: 'error'
      })
      return
    }

    setLoading(true)

    try {
      await authService.resetPassword({
        token: token,
        password: formData.password,
        code: formData.code
      })

      setSnackbar({
        isOpen: true,
        message: t('resetPassword.successMessage'),
        type: 'success'
      })

      // Redirect to login after 2 seconds
      setTimeout(() => {
        navigate('/login')
      }, 2000)
    } catch (err: any) {
      const errorMessage = err.message || t('resetPassword.errorOccurred')
      setSnackbar({
        isOpen: true,
        message: errorMessage,
        type: 'error'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="reset-password-page">
      <Navbar />
      <main className="reset-password-main">
        <div className="reset-password-container">
          <h1 className="reset-password-title">{t('resetPassword.title')}</h1>
          <p className="reset-password-subtitle">{t('resetPassword.subtitle')}</p>

          <form className="reset-password-form" onSubmit={handleSubmit}>
            <Input
              label={t('resetPassword.code')}
              type="text"
              value={formData.code}
              onChange={handleChange('code')}
              placeholder={t('resetPassword.codePlaceholder')}
              required
              error={errors.code}
            />

            <Input
              label={t('resetPassword.newPassword')}
              type="password"
              value={formData.password}
              onChange={handleChange('password')}
              placeholder={t('resetPassword.newPasswordPlaceholder')}
              required
              error={errors.password}
            />

            <Input
              label={t('resetPassword.confirmPassword')}
              type="password"
              value={formData.confirmPassword}
              onChange={handleChange('confirmPassword')}
              placeholder={t('resetPassword.confirmPasswordPlaceholder')}
              required
              error={errors.confirmPassword}
            />

            <button
              type="submit"
              className="reset-password-button"
              disabled={loading || !token}
            >
              {loading ? t('resetPassword.resetting') : t('resetPassword.submit')}
            </button>

            <div className="reset-password-links">
              <Link to="/login" className="reset-password-link">
                {t('resetPassword.backToLogin')}
              </Link>
            </div>
          </form>
        </div>
      </main>

      <Snackbar
        isOpen={snackbar.isOpen}
        message={snackbar.message}
        type={snackbar.type}
        onClose={() => setSnackbar({ ...snackbar, isOpen: false })}
      />
    </div>
  )
}

export default ResetPassword

