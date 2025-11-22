const JWT_USER_KEY = 'jwt_user'

interface DecodedToken {
    user_id: number
    email: string
    exp: number
    iat: number
    jti: string
    token_type: string
    [key: string]: any
}

/**
 * Decode JWT token without verification (client-side only)
 * Note: This doesn't verify the signature, only decodes the payload
 */
function decodeJWT(token: string): DecodedToken | null {
    try {
        const base64Url = token.split('.')[1]
        if (!base64Url) return null

        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/')
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join('')
        )

        return JSON.parse(jsonPayload)
    } catch (error) {
        console.error('Error decoding JWT:', error)
        return null
    }
}

/**
 * Get user ID from JWT token
 */
function getUserIdFromToken(token: string): number | null {
    const decoded = decodeJWT(token)
    return decoded?.user_id || null
}

/**
 * Check if token is expired
 */
function isTokenExpired(token: string): boolean {
    const decoded = decodeJWT(token)
    if (!decoded || !decoded.exp) return true

    const currentTime = Math.floor(Date.now() / 1000)
    return decoded.exp < currentTime
}

export const tokenStorage = {
    /**
     * Set JWT token
     */
    setToken: (token: string) => {
        localStorage.setItem(JWT_USER_KEY, token)
    },

    /**
     * Get JWT token
     */
    getToken: (): string | null => {
        return localStorage.getItem(JWT_USER_KEY)
    },

    /**
     * Remove JWT token
     */
    removeToken: () => {
        localStorage.removeItem(JWT_USER_KEY)
    },

    /**
     * Get user ID from stored token
     */
    getUserId: function (): number | null {
        const token = tokenStorage.getToken()
        if (!token) return null
        return getUserIdFromToken(token)
    },

    /**
     * Get user email from stored token
     */
    getUserEmail: function (): string | null {
        const decoded = tokenStorage.getDecodedToken()
        return decoded?.email || null
    },

    /**
     * Get full user info from token
     */
    getUserInfo: function (): { id: number | null; email: string | null } | null {
        const decoded = tokenStorage.getDecodedToken()
        if (!decoded) return null

        return {
            id: decoded.user_id || null,
            email: decoded.email || null
        }
    },

    /**
     * Get decoded token payload
     */
    getDecodedToken: function (): DecodedToken | null {
        const token = tokenStorage.getToken()
        if (!token) return null
        return decodeJWT(token)
    },

    /**
     * Check if user is authenticated
     */
    isAuthenticated: function (): boolean {
        const token = tokenStorage.getToken()
        if (!token) return false
        return !isTokenExpired(token)
    },

    /**
     * Check if token is expired
     */
    isExpired: function (): boolean {
        const token = tokenStorage.getToken()
        if (!token) return true
        return isTokenExpired(token)
    },

    /**
     * Clear token
     */
    clear: () => {
        localStorage.removeItem(JWT_USER_KEY)
    }
}

