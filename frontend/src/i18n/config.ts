import enTranslations from './en.json'
import trTranslations from './tr.json'

export type SupportedLanguage = 'en' | 'tr'

export const supportedLanguages: SupportedLanguage[] = ['en', 'tr']

export const translations = {
    en: enTranslations,
    tr: trTranslations,
} as const

export const defaultLanguage: SupportedLanguage = 'en'

export const getStoredLanguage = (): SupportedLanguage => {
    if (typeof window === 'undefined') return defaultLanguage
    const stored = localStorage.getItem('i18n_language') as SupportedLanguage
    return stored && supportedLanguages.includes(stored) ? stored : defaultLanguage
}

export const setStoredLanguage = (lang: SupportedLanguage): void => {
    if (typeof window !== 'undefined') {
        localStorage.setItem('i18n_language', lang)
    }
}

