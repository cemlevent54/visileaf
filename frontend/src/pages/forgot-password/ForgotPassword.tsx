import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Navbar from '../../components/navbar/Navbar'
import Input from '../../components/input/Input'
import Snackbar from '../../components/snackbar/Snackbar'
import authService from '../../services/auth.service'
import { useTranslation } from '../../hooks/useTranslation'
import './ForgotPassword.css'

function ForgotPassword() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [email, setEmail] = useState('')
  const [error, setError] = useState('')
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

  const validateEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    return emailRegex.test(email)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')

    // Validation
    if (!email) {
      setError(t('forgotPassword.emailRequired'))
      return
    }

    if (!validateEmail(email)) {
      setError(t('forgotPassword.emailInvalid'))
      return
    }

    setLoading(true)

    try {
      await authService.forgotPassword({ email })
      
      setSnackbar({
        isOpen: true,
        message: t('forgotPassword.successMessage'),
        type: 'success'
      })

      // Redirect to login after 2 seconds
      setTimeout(() => {
        navigate('/login')
      }, 2000)
    } catch (err: any) {
      const errorMessage = err.message || t('forgotPassword.errorOccurred')
      setError(errorMessage)
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
    <div className="forgot-password-page">
      <Navbar />
      <main className="forgot-password-main">
        <div className="forgot-password-container">
          <h1 className="forgot-password-title">{t('forgotPassword.title')}</h1>
          <p className="forgot-password-subtitle">{t('forgotPassword.subtitle')}</p>

          <form className="forgot-password-form" onSubmit={handleSubmit}>
            {error && <div className="error-message">{error}</div>}

            <Input
              label={t('forgotPassword.email')}
              type="email"
              value={email}
              onChange={(e) => {
                setEmail(e.target.value)
                setError('')
              }}
              placeholder={t('forgotPassword.emailPlaceholder')}
              required
              error={error && !email ? error : undefined}
            />

            <button
              type="submit"
              className="forgot-password-button"
              disabled={loading}
            >
              {loading ? t('forgotPassword.sending') : t('forgotPassword.submit')}
            </button>

            <div className="forgot-password-links">
              <Link to="/login" className="forgot-password-link">
                {t('forgotPassword.backToLogin')}
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

export default ForgotPassword

