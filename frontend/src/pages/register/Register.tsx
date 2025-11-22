import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../../components/navbar/Navbar'
import Input from '../../components/input/Input'
import Snackbar from '../../components/snackbar/Snackbar'
import authService from '../../services/auth.service'
import { useTranslation } from '../../hooks/useTranslation'
import './Register.css'

function Register() {
  const navigate = useNavigate()
  const { t } = useTranslation()
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
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
    if (!formData.first_name.trim()) {
      newErrors.first_name = t('register.firstNameRequired')
    }
    if (!formData.last_name.trim()) {
      newErrors.last_name = t('register.lastNameRequired')
    }
    if (!formData.email.trim()) {
      newErrors.email = t('register.emailRequired')
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = t('register.emailInvalid')
    }
    if (!formData.password) {
      newErrors.password = t('register.passwordRequired')
    } else if (formData.password.length < 8) {
      newErrors.password = t('register.passwordMinLength')
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }

    // API call
    setLoading(true)
    setSnackbar({ isOpen: false, message: '', type: 'info' })

    try {
      const response = await authService.register(formData)
      
      if (response.success) {
        // Registration successful
        setSnackbar({
          isOpen: true,
          message: response.message || t('register.registerSuccess'),
          type: 'success'
        })
        // Redirect to login page after showing success message
        setTimeout(() => {
          navigate('/login')
        }, 1500)
      } else {
        setSnackbar({
          isOpen: true,
          message: response.message || t('register.registerFailed'),
          type: 'error'
        })
      }
    } catch (error: any) {
      // Handle API errors
      let errorMessage = t('register.errorOccurred')
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
    <div className="register-page">
      <Navbar />
      <main className="register-main">
        <div className="register-container">
          <h1 className="register-title">{t('register.title')}</h1>
          <p className="register-subtitle">{t('register.subtitle')}</p>
          
          <form className="register-form" onSubmit={handleSubmit}>
            <Input
              label={t('register.firstName')}
              type="text"
              value={formData.first_name}
              onChange={handleChange('first_name')}
              placeholder={t('register.firstNamePlaceholder')}
              required
              error={errors.first_name}
            />
            
            <Input
              label={t('register.lastName')}
              type="text"
              value={formData.last_name}
              onChange={handleChange('last_name')}
              placeholder={t('register.lastNamePlaceholder')}
              required
              error={errors.last_name}
            />
            
            <Input
              label={t('register.email')}
              type="email"
              value={formData.email}
              onChange={handleChange('email')}
              placeholder={t('register.emailPlaceholder')}
              required
              error={errors.email}
            />
            
            <Input
              label={t('register.password')}
              type="password"
              value={formData.password}
              onChange={handleChange('password')}
              placeholder={t('register.passwordPlaceholder')}
              required
              error={errors.password}
            />
            
            <button 
              type="submit" 
              className="register-button"
              disabled={loading}
            >
              {loading ? t('register.registering') : t('register.submit')}
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

export default Register

