import { useLanguage } from '../contexts/LanguageContext'
import type { SupportedLanguage } from '../i18n/config'

/**
 * Translation hook - LanguageContext'i kullanarak çalışır
 * Geriye uyumluluk için useLanguage'in wrapper'ı
 */
export const useTranslation = () => {
    const {
        currentLanguage: language,
        changeLanguage,
        t
    } = useLanguage()

    return {
        t,
        language: language as SupportedLanguage,
        changeLanguage: changeLanguage as (lang: SupportedLanguage) => void,
    }
}

