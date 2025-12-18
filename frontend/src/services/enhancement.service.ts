import { getApiUrl, getAuthHeadersForFormData, apiRequest } from './api.interceptor'

interface EnhancementParams {
    use_gamma?: boolean
    gamma?: number
    use_msr?: boolean
    msr_sigmas?: number[]
    use_clahe?: boolean
    clahe_clip?: number
    clahe_tile_size?: [number, number]
    use_sharpen?: boolean
    sharpen_method?: string
    sharpen_strength?: number
    sharpen_kernel_size?: number
    use_ssr?: boolean
    ssr_sigma?: number
    // Low-light özel alanlar
    use_lowlight_lime?: boolean
    use_lowlight_dual?: boolean
    lowlight_gamma?: number
    lowlight_lambda?: number
    lowlight_sigma?: number
    lowlight_bc?: number
    lowlight_bs?: number
    lowlight_be?: number
    // Eğitimlik temel filtreler
    use_negative?: boolean
    use_threshold?: boolean
    threshold_value?: number
    use_gray_slice?: boolean
    gray_slice_low?: number
    gray_slice_high?: number
    use_bitplane?: boolean
    bitplane_bit?: number
    use_denoise?: boolean
    denoise_strength?: number
    order?: string[]
}

class EnhancementService {
    private baseUrl: string

    constructor() {
        // API base URL doğrudan VITE_API_URL üzerinden gelir (ör. http://localhost:8000)
        this.baseUrl = getApiUrl()
    }

    /**
     * Deep learning tabanlı hazır model ile enhancement (binary response).
     */
    async enhanceImageWithDeepLearning(
        imageFile: File,
        modelName: string
    ): Promise<Blob> {
        const url = `${this.baseUrl}/api/enhancement/with-deep-learning`

        const formData = new FormData()
        formData.append('image', imageFile)
        formData.append('params_json', JSON.stringify({ model_name: modelName }))

        const headers = getAuthHeadersForFormData()

        const response = await apiRequest(url, {
            method: 'POST',
            headers,
            body: formData,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            const errorMessage = errorData.detail || errorData.message || 'Deep learning enhancement failed'
            throw new Error(errorMessage)
        }

        return await response.blob()
    }

    /**
     * Enhance an image with specified parameters
     */
    async enhanceImage(
        imageFile: File,
        params: EnhancementParams
    ): Promise<Blob> {
        try {
            const url = `${this.baseUrl}/api/enhancement/enhance`

            // Create FormData
            const formData = new FormData()
            formData.append('image', imageFile)
            formData.append('params_json', JSON.stringify(params))

            // Get headers for FormData (without Content-Type, browser will set it automatically)
            const headers = getAuthHeadersForFormData()

            const response = await apiRequest(url, {
                method: 'POST',
                headers,
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMessage = errorData.detail || errorData.message || 'Image enhancement failed'
                throw new Error(errorMessage)
            }

            // Return the enhanced image as Blob
            return await response.blob()
        } catch (error) {
            throw error
        }
    }

    /**
     * Enhance an image using Dark Channel Prior (DCP) based low-light enhancement.
     * Optionally accepts the same params JSON as /enhance to control pipeline/order.
     */
    async enhanceImageWithDcp(
        imageFile: File,
        params?: EnhancementParams
    ): Promise<Blob> {
        try {
            const url = `${this.baseUrl}/api/enhancement/enhance-with-dcp`

            const formData = new FormData()
            formData.append('image', imageFile)
            if (params) {
                formData.append('params_json', JSON.stringify(params))
            }

            const headers = getAuthHeadersForFormData()

            const response = await apiRequest(url, {
                method: 'POST',
                headers,
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMessage = errorData.detail || errorData.message || 'Image enhancement with DCP failed'
                throw new Error(errorMessage)
            }

            return await response.blob()
        } catch (error) {
            throw error
        }
    }

    /**
     * Enhance an image using Dark Channel Prior (DCP) + Guided Filter based advanced low-light enhancement.
     * Optionally accepts the same params JSON as /enhance to control pipeline/order.
     */
    async enhanceImageWithDcpGuided(
        imageFile: File,
        params?: EnhancementParams
    ): Promise<Blob> {
        try {
            const url = `${this.baseUrl}/api/enhancement/dcp-guided-filter`

            const formData = new FormData()
            formData.append('image', imageFile)
            if (params) {
                formData.append('params_json', JSON.stringify(params))
            }

            const headers = getAuthHeadersForFormData()

            const response = await apiRequest(url, {
                method: 'POST',
                headers,
                body: formData,
            })

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}))
                const errorMessage = errorData.detail || errorData.message || 'Image enhancement with DCP Guided Filter failed'
                throw new Error(errorMessage)
            }

            return await response.blob()
        } catch (error) {
            throw error
        }
    }

    /**
     * List recent enhancement results for the current user.
     */
    async listResults(): Promise<any[]> {
        const url = `${this.baseUrl}/api/enhancement/results`
        const headers = getAuthHeadersForFormData()

        const response = await apiRequest(url, {
            method: 'GET',
            headers,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            const errorMessage = errorData.detail || errorData.message || 'Failed to load enhancement results'
            throw new Error(errorMessage)
        }

        return await response.json()
    }

    /**
     * Toggle star status for an enhancement result.
     */
    async toggleStar(imageId: string): Promise<{ id: string; is_starred: boolean }> {
        const url = `${this.baseUrl}/api/enhancement/results/${imageId}/toggle-star`
        const headers = getAuthHeadersForFormData()

        const response = await apiRequest(url, {
            method: 'POST',
            headers,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            const errorMessage = errorData.detail || errorData.message || 'Failed to toggle star'
            throw new Error(errorMessage)
        }

        return await response.json()
    }

    /**
     * Hard delete an enhancement result (and its associated files).
     */
    async deleteResult(imageId: string): Promise<void> {
        const url = `${this.baseUrl}/api/enhancement/results/${imageId}`
        const headers = getAuthHeadersForFormData()

        const response = await apiRequest(url, {
            method: 'DELETE',
            headers,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            const errorMessage = errorData.detail || errorData.message || 'Failed to delete result'
            throw new Error(errorMessage)
        }

        // No body expected; deletion successful if we reach here
    }

    /**
     * Export a single enhancement result as PDF (input/output + params).
     */
    async exportResult(imageId: string): Promise<Blob> {
        const url = `${this.baseUrl}/api/enhancement/export/results/${imageId}`
        const headers = getAuthHeadersForFormData()

        const response = await apiRequest(url, {
            method: 'POST',
            headers,
        })

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}))
            const errorMessage = errorData.detail || errorData.message || 'Failed to export result'
            throw new Error(errorMessage)
        }

        return await response.blob()
    }
}

// Export singleton instance
export const enhancementService = new EnhancementService()
export default enhancementService

