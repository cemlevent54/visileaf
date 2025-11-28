import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom'
import { LanguageProvider } from './contexts/LanguageContext'
import Navbar from './components/navbar/Navbar'
import Register from './pages/register/Register'
import Login from './pages/login/Login'
import ForgotPassword from './pages/forgot-password/ForgotPassword'
import ResetPassword from './pages/reset-password/ResetPassword'
import EnhanceYourImage from './pages/enhance-your-image/EnhanceYourImage'
import { useTranslation } from './hooks/useTranslation'
import './App.css'

function Home() {
  const navigate = useNavigate()
  const { t } = useTranslation()

  return (
    <div className="app">
      <Navbar />
      <main className="landing-page">
        <section className="hero">
          <div className="hero-content">
            <h1 className="hero-title">{t('home.title')}</h1>
            <p className="hero-subtitle">
              {t('home.subtitle')}
            </p>
            <div className="hero-buttons">
              <button className="btn-primary" onClick={() => navigate('/register')}>
                {t('home.getStarted')}
              </button>
              <button className="btn-secondary" onClick={() => navigate('/login')}>
                {t('home.login')}
              </button>
            </div>
          </div>
        </section>

        <section className="features">
          <div className="features-container">
            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>{t('home.features.secure.title')}</h3>
              <p>{t('home.features.secure.description')}</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>{t('home.features.fast.title')}</h3>
              <p>{t('home.features.fast.description')}</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>{t('home.features.simple.title')}</h3>
              <p>{t('home.features.simple.description')}</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

function App() {
  return (
    <LanguageProvider>
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/enhance-your-image" element={<EnhanceYourImage />} />
      </Routes>
    </Router>
    </LanguageProvider>
  )
}

export default App
