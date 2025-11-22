import { useState, useEffect, useRef } from 'react'
import { useLanguage } from '../../contexts/LanguageContext'
import { type SupportedLanguage } from '../../i18n/config'
import './LanguageSwitcher.css'

interface LanguageSwitcherProps {
  isMobile?: boolean
  openUpward?: boolean
}

export default function LanguageSwitcher({ isMobile = false, openUpward = false }: LanguageSwitcherProps) {
  const { currentLanguage: language, changeLanguage } = useLanguage()
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  const languages = [
    { 
      code: 'tr' as SupportedLanguage, 
      name: 'TR', 
      flag: '🇹🇷'
    },
    { 
      code: 'en' as SupportedLanguage, 
      name: 'EN', 
      flag: '🇺🇸'
    }
  ]

  const currentLang = languages.find(lang => lang.code === language) || languages[0]

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  const handleLanguageChange = (langCode: SupportedLanguage) => {
    changeLanguage(langCode)
    setIsOpen(false)
  }

  return (
    <div className="language-switcher" ref={dropdownRef}>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="language-switcher-button"
        aria-label="Change language"
        aria-expanded={isOpen}
      >
        <span className="language-switcher-flag">{currentLang.flag}</span>
        <span className="language-switcher-name">{currentLang.name}</span>
        <svg 
          className={`language-switcher-arrow ${isOpen ? 'open' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          <div className="language-switcher-backdrop" onClick={() => setIsOpen(false)} />
          <div className={`language-switcher-dropdown ${isMobile ? 'mobile' : ''} ${openUpward ? 'upward' : ''}`}>
            <div className="language-switcher-list">
              {languages.map((lang) => (
                <button
                  key={lang.code}
                  onClick={() => handleLanguageChange(lang.code)}
                  className={`language-switcher-item ${language === lang.code ? 'active' : ''}`}
                >
                  <span className="language-switcher-item-flag">{lang.flag}</span>
                  <span className="language-switcher-item-name">{lang.name}</span>
                  {language === lang.code && (
                    <svg className="language-switcher-check" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

