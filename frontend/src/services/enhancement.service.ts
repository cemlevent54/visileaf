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
    order?: string[]
}

class EnhancementService {
    private baseUrl: string

    constructor() {
        this.baseUrl = getApiUrl()
        if (!this.baseUrl.startsWith('http://') && !this.baseUrl.startsWith('https://')) {
            this.baseUrl = `http://${this.baseUrl}`
        }
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
}

// Export singleton instance
export const enhancementService = new EnhancementService()
export default enhancementService

