import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import { 
    type SupportedLanguage, 
    getStoredLanguage, 
    setStoredLanguage, 
    defaultLanguage,
    translations 
} from '../i18n/config'

type Language = SupportedLanguage

interface LanguageContextType {
    currentLanguage: Language
    changeLanguage: (lng: Language) => void
    toggleLanguage: () => void
    isTurkish: boolean
    isEnglish: boolean
    isLoaded: boolean
    // Translation function
    t: (key: string) => string
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined)

interface LanguageProviderProps {
    children: ReactNode
}

/**
 * Browser dilini tespit eder
 */
const getBrowserLanguage = (): SupportedLanguage => {
    if (typeof window === 'undefined') return defaultLanguage
    
    const browserLang = navigator.language || (navigator as any).userLanguage
    if (browserLang.startsWith('tr')) return 'tr'
    if (browserLang.startsWith('en')) return 'en'
    
    return defaultLanguage
}

export const LanguageProvider = ({ children }: LanguageProviderProps) => {
    const [currentLanguage, setCurrentLanguage] = useState<Language>(() => {
        // İlk render'da stored language'i kullan, yoksa browser language'i
        if (typeof window !== 'undefined') {
            const stored = getStoredLanguage()
            if (stored !== defaultLanguage) return stored
            return getBrowserLanguage()
        }
        return defaultLanguage
    })
    const [isLoaded, setIsLoaded] = useState(false)

    useEffect(() => {
        // Stored language varsa onu kullan, yoksa browser language'i
        const stored = getStoredLanguage()
        if (stored) {
            setCurrentLanguage(stored)
        } else {
            const browserLang = getBrowserLanguage()
            setCurrentLanguage(browserLang)
            setStoredLanguage(browserLang)
        }
        setIsLoaded(true)
    }, [])

    // Language değiştiğinde localStorage'a kaydet
    useEffect(() => {
        if (isLoaded) {
            setStoredLanguage(currentLanguage)
        }
    }, [currentLanguage, isLoaded])

    const changeLanguage = useCallback((lng: Language) => {
        setCurrentLanguage(lng)
        setStoredLanguage(lng)
        // Context ile tüm bileşenler otomatik güncellenecek
    }, [])

    const toggleLanguage = useCallback(() => {
        const newLang = currentLanguage === 'tr' ? 'en' : 'tr'
        changeLanguage(newLang)
    }, [currentLanguage, changeLanguage])

    // Translation function
    const t = useCallback((key: string): string => {
        const keys = key.split('.')
        let value: any = translations[currentLanguage]

        for (const k of keys) {
            if (value && typeof value === 'object' && k in value) {
                value = value[k]
            } else {
                console.warn(`Translation key not found: ${key}`)
                return key
            }
        }

        return typeof value === 'string' ? value : key
    }, [currentLanguage])

    const value: LanguageContextType = {
        currentLanguage,
        changeLanguage,
        toggleLanguage,
        isTurkish: currentLanguage === 'tr',
        isEnglish: currentLanguage === 'en',
        isLoaded,
        t
    }

    return (
        <LanguageContext.Provider value={value}>
            {children}
        </LanguageContext.Provider>
    )
}

export const useLanguage = () => {
    const context = useContext(LanguageContext)
    if (context === undefined) {
        throw new Error('useLanguage must be used within a LanguageProvider')
    }
    return context
}
