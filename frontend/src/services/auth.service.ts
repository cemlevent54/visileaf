import { tokenStorage } from '../utils/token'
import { apiRequest, getCommonHeaders, getApiUrl } from './api.interceptor'

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

interface ApiResponse<T> {
    success: boolean
    data: T
    message: string
}

interface UserData {
    user: {
        id: number
        first_name: string
        last_name: string
        email: string
        username: string
    }
    tokens?: {
        access: string
        refresh: string
    }
}

class AuthService {
    private baseUrl: string

    constructor() {
        // API interceptor'dan base URL'i al
        this.baseUrl = getApiUrl()

        // Validate URL format
        if (!this.baseUrl.startsWith('http://') && !this.baseUrl.startsWith('https://')) {
            this.baseUrl = `http://${this.baseUrl}`
        }
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
                throw new Error(result.message || 'Registration failed')
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
                throw new Error(result.message || 'Login failed')
            }

            // Store JWT token only
            if (result.success && result.data.tokens) {
                tokenStorage.setToken(result.data.tokens.access)
            }

            return result
        } catch (error) {
            throw error
        }
    }

    /**
     * Logout a user
     */
    logout(): void {
        tokenStorage.clear()
    }

    /**
     * Get current user ID from JWT token
     */
    getCurrentUserId(): number | null {
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
}

// Export singleton instance
export const authService = new AuthService()
export default authService

