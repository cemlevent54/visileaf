import { useState, useEffect } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { tokenStorage } from '../../utils/token'
import { useTranslation } from '../../hooks/useTranslation'
import LanguageSwitcher from '../language-switcher/LanguageSwitcher'
import './Navbar.css'

function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useTranslation()

  useEffect(() => {
    // Check authentication status on mount
    setIsLoggedIn(tokenStorage.isAuthenticated())
  }, [])

  // Close mobile menu when route changes
  useEffect(() => {
    setIsMobileMenuOpen(false)
  }, [location.pathname])

  // Update auth status when route changes
  useEffect(() => {
    setIsLoggedIn(tokenStorage.isAuthenticated())
  }, [location.pathname])

  const handleLogin = () => {
    navigate('/login')
  }

  const handleSignUp = () => {
    navigate('/register')
  }

  const handleLogout = () => {
    tokenStorage.clear()
    setIsLoggedIn(false)
    setIsMobileMenuOpen(false)
    navigate('/')
  }

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false)
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <Link to="/" onClick={closeMobileMenu}>Visileaf</Link>
        </div>
        
        {/* Hamburger Button */}
        <button 
          className={`navbar-hamburger ${isMobileMenuOpen ? 'active' : ''}`}
          onClick={toggleMobileMenu}
          aria-label="Toggle menu"
          aria-expanded={isMobileMenuOpen}
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        {/* Mobile Menu Overlay */}
        {isMobileMenuOpen && (
          <div className="navbar-overlay" onClick={closeMobileMenu}></div>
        )}

        {/* Menu */}
        <ul className={`navbar-menu ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
          <li className="navbar-item">
            <Link to="/" onClick={closeMobileMenu}>{t('navbar.home')}</Link>
          </li>
          <li className="navbar-item">
            <a href="#about" onClick={closeMobileMenu}>{t('navbar.about')}</a>
          </li>
          <li className="navbar-item">
            <a href="#contact" onClick={closeMobileMenu}>{t('navbar.contact')}</a>
          </li>
          <li className="navbar-item navbar-language-selector">
            <LanguageSwitcher isMobile={isMobileMenuOpen} openUpward={false} />
          </li>
          {!isLoggedIn ? (
            <>
              <li className="navbar-item">
                <button className="navbar-button navbar-button-primary" onClick={() => { handleLogin(); closeMobileMenu(); }}>
                  {t('navbar.login')}
                </button>
              </li>
              <li className="navbar-item">
                <button className="navbar-button navbar-button-outline" onClick={() => { handleSignUp(); closeMobileMenu(); }}>
                  {t('navbar.signUp')}
                </button>
              </li>
            </>
          ) : (
            <>
              <li className="navbar-item">
                <button className="navbar-button navbar-button-account">
                  {t('navbar.myAccount')}
                </button>
              </li>
              <li className="navbar-item">
                <button className="navbar-button navbar-button-logout" onClick={() => { handleLogout(); closeMobileMenu(); }}>
                  {t('navbar.logout')}
                </button>
              </li>
            </>
          )}
        </ul>
      </div>
    </nav>
  )
}

export default Navbar

