/**
 * API Request Interceptor
 * Her API request'inde language kontrolü yapar ve otomatik header'lar ekler
 */

import { tokenStorage } from '../utils/token'
import { getStoredLanguage, type SupportedLanguage } from '../i18n/config'

/**
 * API base URL'ini alır (Vite environment variable'dan)
 */
export const getApiUrl = (): string => {
    const envUrl = import.meta.env.VITE_API_URL
    if (envUrl) {
        return envUrl.replace(/\/$/, '')
    }
    return 'http://localhost:8000'
}

/**
 * Language kontrolü yapar ve mevcut dili döndürür
 */
const ensureLanguage = (): SupportedLanguage => {
    if (typeof window === 'undefined') {
        return 'en' // SSR durumunda default
    }

    return getStoredLanguage()
};

/**
 * Ortak header'ları oluşturur
 */
export const getCommonHeaders = (): Record<string, string> => {
    const language = ensureLanguage();

    return {
        'Content-Type': 'application/json',
        'Accept-Language': language
    };
};

/**
 * Auth header'ları ile birlikte ortak header'ları oluşturur
 */
export const getAuthHeaders = (): Record<string, string> => {
    const token = tokenStorage.getToken();
    const language = ensureLanguage();

    console.log('🔑 getAuthHeaders - Token:', token ? 'Present' : 'Missing', 'Length:', token?.length || 0);

    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        'Accept-Language': language
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
};

/**
 * FormData için auth header'ları oluşturur (Content-Type olmadan)
 */
export const getAuthHeadersForFormData = (): Record<string, string> => {
    const token = tokenStorage.getToken();
    const language = ensureLanguage();

    const headers: Record<string, string> = {
        'Accept-Language': language
    };

    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    return headers;
};

/**
 * API request wrapper - otomatik language kontrolü ile
 */
export const apiRequest = async (
    url: string,
    options: RequestInit = {}
): Promise<Response> => {
    const language = ensureLanguage();

    // Mevcut headers'ı koru ve Accept-Language ekle
    const headers = {
        ...options.headers,
        'Accept-Language': language
    };

    console.log(`🔍 API Request - URL: ${url}, Language: ${language}`);

    const response = await fetch(url, {
        ...options,
        headers
    });

    // 401 Unauthorized kontrolü - token süresi dolmuş
    // Sadece login sayfası değilse yönlendir (login sayfasında snackbar gösterilecek)
    if (response.status === 401 && typeof window !== 'undefined' && window.location.pathname !== '/login') {
        console.warn('🚨 401 Unauthorized - Token expired, redirecting to login');

        // Token'ı temizle
        tokenStorage.clear();

        // Auth state değişikliği event'i gönder
        window.dispatchEvent(new CustomEvent('authStateChanged'));

        // Login sayfasına yönlendir
        window.location.href = '/login';

        // Response'u döndür (component'ler hata handling yapabilir)
        return response;
    }

    return response;
};