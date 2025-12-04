import { tokenStorage } from '../utils/token'
import { apiRequest, getCommonHeaders, getAuthHeaders, getApiUrl } from './api.interceptor'

interface RegisterData {
    first_name: string
    last_name: string
    email: string
    password: string
}

interface LoginData {
    email: string
    password: string
}

interface ForgotPasswordData {
    email: string
}

interface ResetPasswordData {
    token: string
    password: string
    code: string
}

interface MeResponse {
    success: boolean
    data: {
        user: {
            id: string
            first_name: string | null
            last_name: string | null
            email: string
        }
    }
    message: string
}

interface ApiResponse<T> {
    success: boolean
    data: T
    message: string
}

interface UserData {
    user: {
        id: string  // UUID from backend
        first_name: string | null
        last_name: string | null
        email: string
    }
    tokens?: {
        access: string
        refresh: string
    }
}

class AuthService {
    private baseUrl: string

    constructor() {
        // API base URL doğrudan VITE_API_URL üzerinden gelir (ör. http://localhost:8000)
        this.baseUrl = getApiUrl()
    }

    /**
     * Register a new user
     */
    async register(data: RegisterData): Promise<ApiResponse<UserData>> {
        try {
            const url = `${this.baseUrl}/api/auth/register`
            const response = await apiRequest(url, {
                method: 'POST',
                headers: getCommonHeaders(),
                body: JSON.stringify(data),
            })

            const result = await response.json()

            if (!response.ok) {
                // Backend returns error in 'detail' field for HTTPException
                const errorMessage = result.detail || result.message || 'Registration failed'
                throw new Error(errorMessage)
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Login a user
     */
    async login(data: LoginData): Promise<ApiResponse<UserData>> {
        try {
            const url = `${this.baseUrl}/api/auth/login`
            const response = await apiRequest(url, {
                method: 'POST',
                headers: getCommonHeaders(),
                body: JSON.stringify(data),
            })

            const result = await response.json()

            if (!response.ok) {
                // Backend returns error in 'detail' field for HTTPException
                const errorMessage = result.detail || result.message || 'Login failed'
                throw new Error(errorMessage)
            }

            // Store JWT tokens
            if (result.success && result.data?.tokens) {
                tokenStorage.setToken(result.data.tokens.access)
                // Store refresh token in localStorage
                if (result.data.tokens.refresh) {
                    localStorage.setItem('refresh_token', result.data.tokens.refresh)
                }
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Logout a user
     */
    async logout(refreshToken?: string): Promise<void> {
        try {
            // Get refresh token from localStorage if not provided
            const token = refreshToken || localStorage.getItem('refresh_token')

            // If refresh token exists, call backend logout endpoint
            if (token) {
                const url = `${this.baseUrl}/api/auth/logout`
                try {
                    await apiRequest(url, {
                        method: 'POST',
                        headers: getCommonHeaders(),
                        body: JSON.stringify({ refresh_token: token }),
                    })
                } catch (error) {
                    // Even if backend logout fails, clear local token
                    console.warn('Backend logout failed, clearing local token:', error)
                }
            }
        } catch (error) {
            // Even if logout fails, clear local token
            console.warn('Logout error:', error)
        } finally {
            // Always clear local tokens
            tokenStorage.clear()
            localStorage.removeItem('refresh_token')
        }
    }

    /**
     * Get current user ID from JWT token
     */
    getCurrentUserId(): string | null {
        return tokenStorage.getUserId()
    }

    /**
     * Get decoded token payload
     */
    getDecodedToken(): any {
        return tokenStorage.getDecodedToken()
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated(): boolean {
        return tokenStorage.isAuthenticated()
    }

    /**
     * Get current user information
     */
    async getMe(): Promise<MeResponse> {
        try {
            const url = `${this.baseUrl}/api/auth/me`
            const token = tokenStorage.getToken()

            if (!token) {
                throw new Error('No authentication token found')
            }

            const response = await apiRequest(url, {
                method: 'GET',
                headers: getAuthHeaders(),
            })

            const result = await response.json()

            if (!response.ok) {
                const errorMessage = result.detail || result.message || 'Failed to get user information'
                throw new Error(errorMessage)
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Refresh access token
     */
    async refreshToken(refreshToken?: string): Promise<ApiResponse<{ access: string }>> {
        try {
            // Get refresh token from localStorage if not provided
            const token = refreshToken || localStorage.getItem('refresh_token')

            if (!token) {
                throw new Error('No refresh token found')
            }

            const url = `${this.baseUrl}/api/auth/refresh-token`
            const response = await apiRequest(url, {
                method: 'POST',
                headers: getCommonHeaders(),
                body: JSON.stringify({ refresh_token: token }),
            })

            const result = await response.json()

            if (!response.ok) {
                const errorMessage = result.detail || result.message || 'Token refresh failed'
                throw new Error(errorMessage)
            }

            // Store new access token
            if (result.success && result.data?.access) {
                tokenStorage.setToken(result.data.access)
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Request password reset token
     */
    async forgotPassword(data: ForgotPasswordData): Promise<ApiResponse<{ user: any }>> {
        try {
            const url = `${this.baseUrl}/api/auth/forgot-password`
            const response = await apiRequest(url, {
                method: 'POST',
                headers: getCommonHeaders(),
                body: JSON.stringify(data),
            })

            const result = await response.json()

            if (!response.ok) {
                const errorMessage = result.detail || result.message || 'Failed to process forgot password request'
                throw new Error(errorMessage)
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Reset password using reset token
     */
    async resetPassword(data: ResetPasswordData): Promise<ApiResponse<UserData>> {
        try {
            const url = `${this.baseUrl}/api/auth/reset-password`
            const response = await apiRequest(url, {
                method: 'POST',
                headers: getCommonHeaders(),
                body: JSON.stringify(data),
            })

            const result = await response.json()

            if (!response.ok) {
                const errorMessage = result.detail || result.message || 'Password reset failed'
                throw new Error(errorMessage)
            }

            return result
        } catch (error) {
            throw error
        }
    }
}

// Export singleton instance
export const authService = new AuthService()
export default authService

