import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../../components/navbar/Navbar'
import Input from '../../components/input/Input'
import Snackbar from '../../components/snackbar/Snackbar'
import authService from '../../services/auth.service'
import { useTranslation } from '../../hooks/useTranslation'
import './Login.css'

function Login() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [formData, setFormData] = useState({
    email: '',
    password: ''
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

  const handleChange = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value
    }))
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Reset errors
    setErrors({})

    // Basic validation
    const newErrors: Record<string, string> = {}
    if (!formData.email.trim()) {
      newErrors.email = t('login.emailRequired')
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = t('login.emailInvalid')
    }
    if (!formData.password) {
      newErrors.password = t('login.passwordRequired')
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }

    // API call
    setLoading(true)
    setSnackbar({ isOpen: false, message: '', type: 'info' })

    try {
      const response = await authService.login(formData)
      
      if (response.success) {
        // Login successful - tokens are already stored in authService.login()
        setSnackbar({
          isOpen: true,
          message: response.message || t('login.loginSuccess'),
          type: 'success'
        })
        // Redirect to home page after showing success message
        setTimeout(() => {
          navigate('/')
        }, 1500)
      } else {
        setSnackbar({
          isOpen: true,
          message: response.message || t('login.loginFailed'),
          type: 'error'
        })
      }
    } catch (error: any) {
      // Handle API errors
      let errorMessage = t('login.errorOccurred')
      if (error.message) {
        errorMessage = error.message
      } else if (typeof error === 'string') {
        errorMessage = error
      }
      
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
    <div className="login-page">
      <Navbar />
      <main className="login-main">
        <div className="login-container">
          <h1 className="login-title">{t('login.title')}</h1>
          <p className="login-subtitle">{t('login.subtitle')}</p>
          
          <form className="login-form" onSubmit={handleSubmit}>
            <Input
              label={t('login.email')}
              type="email"
              value={formData.email}
              onChange={handleChange('email')}
              placeholder={t('login.emailPlaceholder')}
              required
              error={errors.email}
            />
            
            <Input
              label={t('login.password')}
              type="password"
              value={formData.password}
              onChange={handleChange('password')}
              placeholder={t('login.passwordPlaceholder')}
              required
              error={errors.password}
            />
            
            <button 
              type="submit" 
              className="login-button"
              disabled={loading}
            >
              {loading ? t('login.loggingIn') : t('login.submit')}
            </button>
          </form>
        </div>
      </main>
      
      <Snackbar
        message={snackbar.message}
        type={snackbar.type}
        isOpen={snackbar.isOpen}
        onClose={() => setSnackbar({ ...snackbar, isOpen: false })}
      />
    </div>
  )
}

export default Login

