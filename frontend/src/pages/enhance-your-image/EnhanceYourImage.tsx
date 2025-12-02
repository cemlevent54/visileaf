import { useState, useRef } from 'react'
import Navbar from '../../components/navbar/Navbar'
import Snackbar from '../../components/snackbar/Snackbar'
import Checkbox from '../../components/checkbox/Checkbox'
import RadioButton from '../../components/radio-button/RadioButton'
import VolumeSlider from '../../components/volume-slider/VolumeSlider'
import Input from '../../components/input/Input'
import enhancementService from '../../services/enhancement.service'
import { useTranslation } from '../../hooks/useTranslation'
import './EnhanceYourImage.css'

interface EnhancementParams {
  use_gamma: boolean
  gamma: number
  use_msr: boolean
  msr_sigmas: [number, number, number]
  use_clahe: boolean
  clahe_clip: number
  clahe_tile_size: [number, number]
  use_sharpen: boolean
  sharpen_method: 'unsharp' | 'laplacian'
  sharpen_strength: number
  sharpen_kernel_size: number
  use_ssr: boolean
  ssr_sigma: number
  // Eğitimlik temel filtreler
  use_negative: boolean
  use_threshold: boolean
  threshold_value: number
  use_gray_slice: boolean
  gray_slice_low: number
  gray_slice_high: number
  use_bitplane: boolean
  bitplane_bit: number
  // Low-light özel modları
  use_lowlight_lime: boolean
  use_lowlight_dual: boolean
  lowlight_gamma: number
  lowlight_lambda: number
  lowlight_sigma: number
  lowlight_bc: number
  lowlight_bs: number
  lowlight_be: number
  order: string[]
}

function EnhanceYourImage() {
  const { t } = useTranslation()
  const [inputImage, setInputImage] = useState<File | null>(null)
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null)
  const [outputImage, setOutputImage] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null)

  // Enhancement parameters state
  const [params, setParams] = useState<EnhancementParams>({
    use_gamma: true,
    gamma: 0.5,
    use_msr: true,
    msr_sigmas: [15, 80, 250],
    use_clahe: true,
    clahe_clip: 2.5,
    clahe_tile_size: [8, 8],
    use_sharpen: true,
    sharpen_method: 'unsharp',
    sharpen_strength: 1.5,
    sharpen_kernel_size: 5,
    use_ssr: false,
    ssr_sigma: 80,
    // Eğitimlik temel filtreler (varsayılan: kapalı)
    use_negative: false,
    use_threshold: false,
    threshold_value: 128,
    use_gray_slice: false,
    gray_slice_low: 100,
    gray_slice_high: 180,
    use_bitplane: false,
    bitplane_bit: 7,
    // Low-light default parametreler (backend ile uyumlu)
    use_lowlight_lime: false,
    use_lowlight_dual: false,
    lowlight_gamma: 0.6,
    lowlight_lambda: 0.15,
    lowlight_sigma: 3.0,
    lowlight_bc: 1.0,
    lowlight_bs: 1.0,
    lowlight_be: 1.0,
    order: ['gamma', 'msr', 'clahe', 'sharpen']
  })

  const [snackbar, setSnackbar] = useState<{
    isOpen: boolean
    message: string
    type: 'success' | 'error' | 'info' | 'warning'
  }>({
    isOpen: false,
    message: '',
    type: 'info'
  })

  // Get active methods
  const getActiveMethods = (): string[] => {
    const activeMethods: string[] = []
    if (params.use_gamma) activeMethods.push('gamma')
    if (params.use_msr) activeMethods.push('msr')
    if (params.use_clahe) activeMethods.push('clahe')
    if (params.use_sharpen) activeMethods.push('sharpen')
    if (params.use_ssr) activeMethods.push('ssr')
    if (params.use_negative) activeMethods.push('negative')
    if (params.use_threshold) activeMethods.push('threshold')
    if (params.use_gray_slice) activeMethods.push('gray_slice')
    if (params.use_bitplane) activeMethods.push('bitplane')
    if (params.use_lowlight_lime) activeMethods.push('lowlight_lime')
    if (params.use_lowlight_dual) activeMethods.push('lowlight_dual')
    return activeMethods
  }

  // Update order based on active methods (only if order is empty or invalid)
  const updateOrder = () => {
    const activeMethods = getActiveMethods()
    
    // Keep existing order if it's valid, otherwise update
    const currentOrder = params.order.filter(m => activeMethods.includes(m))
    const missingMethods = activeMethods.filter(m => !currentOrder.includes(m))
    const newOrder = [...currentOrder, ...missingMethods]
    
    setParams(prev => ({ ...prev, order: newOrder }))
  }

  // Handle order drag start
  const handleDragStart = (index: number) => {
    setDraggedIndex(index)
  }

  // Handle order drag over
  const handleDragOverOrder = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    if (draggedIndex === null || draggedIndex === index) return

    const newOrder = [...finalOrder]
    const draggedItem = newOrder[draggedIndex]
    newOrder.splice(draggedIndex, 1)
    newOrder.splice(index, 0, draggedItem)
    
    setParams(prev => ({ ...prev, order: newOrder }))
    setDraggedIndex(index)
  }

  // Handle order drag end
  const handleDragEnd = () => {
    setDraggedIndex(null)
  }

  // Get method display name
  const getMethodName = (method: string): string => {
    return t(`enhance.methodNames.${method}`) || method
  }

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.type.startsWith('image/')) {
        setSnackbar({
          isOpen: true,
          message: t('enhance.invalidFileType'),
          type: 'error'
        })
        return
      }

      if (file.size > 10 * 1024 * 1024) {
        setSnackbar({
          isOpen: true,
          message: t('enhance.fileTooLarge'),
          type: 'error'
        })
        return
      }

      setInputImage(file)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setInputImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
      
      setOutputImage(null)
    }
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      if (!file.type.startsWith('image/')) {
        setSnackbar({
          isOpen: true,
          message: t('enhance.invalidFileType'),
          type: 'error'
        })
        return
      }

      if (file.size > 10 * 1024 * 1024) {
        setSnackbar({
          isOpen: true,
          message: t('enhance.fileTooLarge'),
          type: 'error'
        })
        return
      }

      setInputImage(file)
      
      const reader = new FileReader()
      reader.onloadend = () => {
        setInputImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
      
      setOutputImage(null)
    }
  }

  const handleDragOverImage = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleEnhance = async () => {
    if (!inputImage) {
      setSnackbar({
        isOpen: true,
        message: t('enhance.noImageSelected'),
        type: 'warning'
      })
      return
    }

    setLoading(true)
    setSnackbar({ isOpen: false, message: '', type: 'info' })

    try {
      // Prepare parameters for API
      const apiParams: any = {
        use_gamma: params.use_gamma,
        gamma: params.gamma,
        use_msr: params.use_msr,
        msr_sigmas: params.msr_sigmas,
        use_clahe: params.use_clahe,
        clahe_clip: params.clahe_clip,
        clahe_tile_size: params.clahe_tile_size,
        use_sharpen: params.use_sharpen,
        sharpen_method: params.sharpen_method,
        sharpen_strength: params.sharpen_strength,
        sharpen_kernel_size: params.sharpen_kernel_size,
        use_ssr: params.use_ssr,
        ssr_sigma: params.ssr_sigma,
        // Eğitimlik temel filtreler
        use_negative: params.use_negative,
        use_threshold: params.use_threshold,
        threshold_value: params.threshold_value,
        use_gray_slice: params.use_gray_slice,
        gray_slice_low: params.gray_slice_low,
        gray_slice_high: params.gray_slice_high,
        use_bitplane: params.use_bitplane,
        bitplane_bit: params.bitplane_bit,
        // Low-light parametreleri
        use_lowlight_lime: params.use_lowlight_lime,
        use_lowlight_dual: params.use_lowlight_dual,
        lowlight_gamma: params.lowlight_gamma,
        lowlight_lambda: params.lowlight_lambda,
        lowlight_sigma: params.lowlight_sigma,
        lowlight_bc: params.lowlight_bc,
        lowlight_bs: params.lowlight_bs,
        lowlight_be: params.lowlight_be,
        order: params.order
      }

      const enhancedImageBlob = await enhancementService.enhanceImage(inputImage, apiParams)
      
      const imageUrl = URL.createObjectURL(enhancedImageBlob)
      setOutputImage(imageUrl)

      setSnackbar({
        isOpen: true,
        message: t('enhance.success'),
        type: 'success'
      })
    } catch (error: any) {
      const errorMessage = error.message || t('enhance.errorOccurred')
      setSnackbar({
        isOpen: true,
        message: errorMessage,
        type: 'error'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = () => {
    if (outputImage) {
      const link = document.createElement('a')
      link.href = outputImage
      link.download = 'enhanced_image.jpg'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const handleClear = () => {
    setInputImage(null)
    setInputImagePreview(null)
    setOutputImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const activeMethods = getActiveMethods()
  const displayOrder = params.order.filter(m => activeMethods.includes(m))
  const missingInOrder = activeMethods.filter(m => !displayOrder.includes(m))
  const finalOrder = [...displayOrder, ...missingInOrder]

  return (
    <div className="enhance-page">
      <Navbar />
      <main className="enhance-main">
        <div className="enhance-container">
          <h1 className="enhance-title">{t('enhance.title')}</h1>
          <p className="enhance-subtitle">{t('enhance.subtitle')}</p>

          <div className="enhance-layout">
            {/* Left Column - Images */}
            <div className="enhance-images-column">
              {/* Input Image Section */}
              <div className="image-section">
                <h2 className="section-title">{t('enhance.inputImage')}</h2>
                <div
                  className="image-upload-area"
                  onDrop={handleDrop}
                  onDragOver={handleDragOverImage}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    style={{ display: 'none' }}
                  />
                  {inputImagePreview ? (
                    <div className="image-preview-container">
                      <img
                        src={inputImagePreview}
                        alt="Input preview"
                        className="image-preview"
                      />
                      <button
                        className="clear-button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleClear()
                        }}
                        aria-label="Clear image"
                      >
                        ×
                      </button>
                    </div>
                  ) : (
                    <div className="upload-placeholder">
                      <div className="upload-icon">📷</div>
                      <p className="upload-text">{t('enhance.uploadText')}</p>
                      <p className="upload-hint">{t('enhance.uploadHint')}</p>
                    </div>
                  )}
                </div>
                {inputImage && (
                  <p className="image-info">
                    {inputImage.name} ({(inputImage.size / 1024 / 1024).toFixed(2)} MB)
                  </p>
                )}
              </div>

              {/* Output Image Section */}
              <div className="image-section">
                <h2 className="section-title">{t('enhance.outputImage')}</h2>
                <div className="image-display-area">
                  {outputImage ? (
                    <div className="image-preview-container">
                      <img
                        src={outputImage}
                        alt="Enhanced output"
                        className="image-preview"
                      />
                      <button
                        className="download-button"
                        onClick={handleDownload}
                        aria-label="Download image"
                      >
                        ⬇️ {t('enhance.download')}
                      </button>
                    </div>
                  ) : (
                    <div className="output-placeholder">
                      <div className="placeholder-icon">✨</div>
                      <p className="placeholder-text">{t('enhance.outputPlaceholder')}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Action Button */}
              <div className="action-section">
                <button
                  className="enhance-button"
                  onClick={handleEnhance}
                  disabled={!inputImage || loading}
                >
                  {loading ? t('enhance.processing') : t('enhance.enhanceButton')}
                </button>
              </div>
            </div>

            {/* Right Column - Parameters */}
            <div className="enhance-parameters-column">
              <h2 className="parameters-title">{t('enhance.parameters')}</h2>
              
              <div className="parameters-content">
                {/* Gamma Correction */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.gammaCorrection')}</h3>
                  <Checkbox
                    label={t('enhance.useGamma')}
                    name="use_gamma"
                    value={params.use_gamma ? 'checked' : ''}
                    onChange={(value) => {
                      setParams(prev => ({ ...prev, use_gamma: value === 'checked' }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_gamma && (
                    <VolumeSlider
                      label={t('enhance.gammaValue')}
                      value={params.gamma}
                      onChange={(value) => setParams(prev => ({ ...prev, gamma: value }))}
                      min={0.1}
                      max={2.0}
                      step={0.1}
                      showValue={true}
                    />
                  )}
                </div>

                {/* CLAHE */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.clahe')}</h3>
                  <Checkbox
                    label={t('enhance.useClahe')}
                    name="use_clahe"
                    value={params.use_clahe ? 'checked' : ''}
                    onChange={(value) => {
                      setParams(prev => ({ ...prev, use_clahe: value === 'checked' }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_clahe && (
                    <>
                      <VolumeSlider
                        label={t('enhance.claheClip')}
                        value={params.clahe_clip}
                        onChange={(value) => setParams(prev => ({ ...prev, clahe_clip: value }))}
                        min={0.1}
                        max={5.0}
                        step={0.1}
                        showValue={true}
                      />
                      <div className="tile-size-inputs">
                        <Input
                          label={t('enhance.tileWidth')}
                          type="number"
                          value={params.clahe_tile_size[0].toString()}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) || 8
                            setParams(prev => ({ 
                              ...prev, 
                              clahe_tile_size: [val, prev.clahe_tile_size[1]] 
                            }))
                          }}
                          placeholder="8"
                        />
                        <Input
                          label={t('enhance.tileHeight')}
                          type="number"
                          value={params.clahe_tile_size[1].toString()}
                          onChange={(e) => {
                            const val = parseInt(e.target.value) || 8
                            setParams(prev => ({ 
                              ...prev, 
                              clahe_tile_size: [prev.clahe_tile_size[0], val] 
                            }))
                          }}
                          placeholder="8"
                        />
                      </div>
                    </>
                  )}
                </div>

                {/* MSR */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.msr')}</h3>
                  <Checkbox
                    label={t('enhance.useMsr')}
                    name="use_msr"
                    value={params.use_msr ? 'checked' : ''}
                    onChange={(value) => {
                      setParams(prev => ({ ...prev, use_msr: value === 'checked' }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_msr && (
                    <div className="msr-sigmas-inputs">
                      <Input
                        label={t('enhance.sigma1')}
                        type="number"
                        value={params.msr_sigmas[0].toString()}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 15
                          setParams(prev => ({ 
                            ...prev, 
                            msr_sigmas: [val, prev.msr_sigmas[1], prev.msr_sigmas[2]] 
                          }))
                        }}
                        placeholder="15"
                      />
                      <Input
                        label={t('enhance.sigma2')}
                        type="number"
                        value={params.msr_sigmas[1].toString()}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 80
                          setParams(prev => ({ 
                            ...prev, 
                            msr_sigmas: [prev.msr_sigmas[0], val, prev.msr_sigmas[2]] 
                          }))
                        }}
                        placeholder="80"
                      />
                      <Input
                        label={t('enhance.sigma3')}
                        type="number"
                        value={params.msr_sigmas[2].toString()}
                        onChange={(e) => {
                          const val = parseInt(e.target.value) || 250
                          setParams(prev => ({ 
                            ...prev, 
                            msr_sigmas: [prev.msr_sigmas[0], prev.msr_sigmas[1], val] 
                          }))
                        }}
                        placeholder="250"
                      />
                    </div>
                  )}
                </div>

                {/* SSR */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.ssr')}</h3>
                  <Checkbox
                    label={t('enhance.useSsr')}
                    name="use_ssr"
                    value={params.use_ssr ? 'checked' : ''}
                    onChange={(value) => {
                      setParams(prev => ({ ...prev, use_ssr: value === 'checked' }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_ssr && (
                    <Input
                      label={t('enhance.ssrSigma')}
                      type="number"
                      value={params.ssr_sigma.toString()}
                      onChange={(e) => {
                        const val = parseInt(e.target.value) || 80
                        setParams(prev => ({ ...prev, ssr_sigma: val }))
                      }}
                      placeholder="80"
                    />
                  )}
                </div>

                {/* Sharpen */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.sharpen')}</h3>
                  <Checkbox
                    label={t('enhance.useSharpen')}
                    name="use_sharpen"
                    value={params.use_sharpen ? 'checked' : ''}
                    onChange={(value) => {
                      setParams(prev => ({ ...prev, use_sharpen: value === 'checked' }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_sharpen && (
                    <>
                      <RadioButton
                        label={t('enhance.sharpenMethod')}
                        name="sharpen_method"
                        options={[
                          { value: 'unsharp', label: t('enhance.unsharp') },
                          { value: 'laplacian', label: t('enhance.laplacian') }
                        ]}
                        value={params.sharpen_method}
                        onChange={(value) => setParams(prev => ({ 
                          ...prev, 
                          sharpen_method: value as 'unsharp' | 'laplacian' 
                        }))}
                        inline={true}
                      />
                      <VolumeSlider
                        label={t('enhance.sharpenStrength')}
                        value={params.sharpen_strength}
                        onChange={(value) => setParams(prev => ({ ...prev, sharpen_strength: value }))}
                        min={0.1}
                        max={3.0}
                        step={0.1}
                        showValue={true}
                      />
                      <Input
                        label={`${t('enhance.sharpenKernelSize')} (${t('enhance.sharpenKernelSizeHint')})`}
                        type="number"
                        value={params.sharpen_kernel_size.toString()}
                        onChange={(e) => {
                          const inputVal = e.target.value
                          if (inputVal === '') {
                            setParams(prev => ({ ...prev, sharpen_kernel_size: 5 }))
                            return
                          }
                          
                          let val = parseInt(inputVal) || 5
                          
                          // Min/Max kontrolü
                          if (val < 1) val = 1
                          if (val > 21) val = 21
                          
                          // Tek sayı kontrolü - çift sayıysa en yakın tek sayıya yuvarla
                          if (val % 2 === 0) {
                            // Çift sayı girildi, en yakın tek sayıya yuvarla (yukarı)
                            val = val + 1
                            if (val > 21) val = val - 2 // Yukarı çıkamazsa aşağı al
                            
                            // Kullanıcıya bildir
                            setSnackbar({
                              isOpen: true,
                              message: t('enhance.kernelSizeInvalid') + ` → ${val}`,
                              type: 'warning'
                            })
                          }
                          
                          setParams(prev => ({ ...prev, sharpen_kernel_size: val }))
                        }}
                        placeholder="5"
                        min={1}
                        max={21}
                        step={2}
                      />
                    </>
                  )}
                </div>

                {/* Educational / Basic Filters */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">
                    {t('enhance.educationalFilters') || 'Educational Filters'}
                  </h3>
                  <p className="order-hint">
                    {t('enhance.educationalFiltersHint') ||
                      'Simple teaching-oriented filters: negative, thresholding, gray-level slicing, bit-plane slicing.'}
                  </p>
                  <Checkbox
                    label={t('enhance.useNegative') || 'Use Negative Image'}
                    name="use_negative"
                    value={params.use_negative ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_negative: checked
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  <Checkbox
                    label={t('enhance.useThreshold') || 'Use Binary Threshold'}
                    name="use_threshold"
                    value={params.use_threshold ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_threshold: checked
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_threshold && (
                    <VolumeSlider
                      label={t('enhance.thresholdValue') || 'Threshold Value'}
                      value={params.threshold_value}
                      onChange={(value) =>
                        setParams((prev) => ({
                          ...prev,
                          threshold_value: Math.round(value)
                        }))
                      }
                      min={0}
                      max={255}
                      step={1}
                      showValue={true}
                    />
                  )}
                  <Checkbox
                    label={t('enhance.useGraySlice') || 'Use Gray-level Slicing'}
                    name="use_gray_slice"
                    value={params.use_gray_slice ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_gray_slice: checked
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_gray_slice && (
                    <div className="msr-sigmas-inputs">
                      <Input
                        label={t('enhance.graySliceLow') || 'Gray slice low'}
                        type="number"
                        value={params.gray_slice_low.toString()}
                        onChange={(e) => {
                          let val = parseInt(e.target.value) || 0
                          if (val < 0) val = 0
                          if (val > 255) val = 255
                          setParams((prev) => ({
                            ...prev,
                            gray_slice_low: val
                          }))
                        }}
                        placeholder="100"
                      />
                      <Input
                        label={t('enhance.graySliceHigh') || 'Gray slice high'}
                        type="number"
                        value={params.gray_slice_high.toString()}
                        onChange={(e) => {
                          let val = parseInt(e.target.value) || 0
                          if (val < 0) val = 0
                          if (val > 255) val = 255
                          setParams((prev) => ({
                            ...prev,
                            gray_slice_high: val
                          }))
                        }}
                        placeholder="180"
                      />
                    </div>
                  )}
                  <Checkbox
                    label={t('enhance.useBitplane') || 'Use Bit-plane Slicing'}
                    name="use_bitplane"
                    value={params.use_bitplane ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_bitplane: checked
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  {params.use_bitplane && (
                    <VolumeSlider
                      label={t('enhance.bitplaneBit') || 'Bit-plane bit (0-7)'}
                      value={params.bitplane_bit}
                      onChange={(value) =>
                        setParams((prev) => ({
                          ...prev,
                          bitplane_bit: Math.round(Math.min(7, Math.max(0, value)))
                        }))
                      }
                      min={0}
                      max={7}
                      step={1}
                      showValue={true}
                    />
                  )}
                </div>

                {/* Low-light (LIME / DUAL approx) */}
                <div className="parameter-group">
                  <h3 className="parameter-group-title">{t('enhance.lowlightTitle') || 'Low-light Enhancement'}</h3>
                  <p className="order-hint">
                    {t('enhance.lowlightHint') ||
                      'Low-light preset uses MSR + CLAHE + Gamma to approximate LIME/DUAL behaviour.'}
                  </p>
                  <Checkbox
                    label={t('enhance.useLowlightLime') || 'Use Low-light (LIME-like)'}
                    name="use_lowlight_lime"
                    value={params.use_lowlight_lime ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_lowlight_lime: checked,
                        // Aynı anda hem LIME hem DUAL aktif olmasın; kullanıcı isterse elle açabilir
                        use_lowlight_dual: checked ? false : prev.use_lowlight_dual
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />
                  <Checkbox
                    label={t('enhance.useLowlightDual') || 'Use Low-light (DUAL-like)'}
                    name="use_lowlight_dual"
                    value={params.use_lowlight_dual ? 'checked' : ''}
                    onChange={(value) => {
                      const checked = value === 'checked'
                      setParams((prev) => ({
                        ...prev,
                        use_lowlight_dual: checked,
                        // Aynı anda hem LIME hem DUAL aktif olmasın; kullanıcı isterse elle açabilir
                        use_lowlight_lime: checked ? false : prev.use_lowlight_lime
                      }))
                      setTimeout(updateOrder, 0)
                    }}
                    single={true}
                  />

                  {(params.use_lowlight_lime || params.use_lowlight_dual) && (
                    <>
                      <VolumeSlider
                        label={t('enhance.lowlightGamma') || 'Low-light Gamma'}
                        value={params.lowlight_gamma}
                        onChange={(value) =>
                          setParams((prev) => ({
                            ...prev,
                            lowlight_gamma: value
                          }))
                        }
                        min={0.1}
                        max={2.0}
                        step={0.1}
                        showValue={true}
                      />
                      <VolumeSlider
                        label={t('enhance.lowlightLambda') || 'Low-light λ (lambda)'}
                        value={params.lowlight_lambda}
                        onChange={(value) =>
                          setParams((prev) => ({
                            ...prev,
                            lowlight_lambda: value
                          }))
                        }
                        min={0.01}
                        max={1.0}
                        step={0.01}
                        showValue={true}
                      />
                      <VolumeSlider
                        label={t('enhance.lowlightSigma') || 'Low-light σ (sigma)'}
                        value={params.lowlight_sigma}
                        onChange={(value) =>
                          setParams((prev) => ({
                            ...prev,
                            lowlight_sigma: value
                          }))
                        }
                        min={0.5}
                        max={10.0}
                        step={0.5}
                        showValue={true}
                      />
                      <div className="msr-sigmas-inputs">
                        <Input
                          label={t('enhance.lowlightBc') || 'Contrast weight (bc)'}
                          type="number"
                          value={params.lowlight_bc.toString()}
                          step={0.1}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value)
                            setParams((prev) => ({
                              ...prev,
                              lowlight_bc: isNaN(val) ? 1.0 : val
                            }))
                          }}
                          placeholder="1.0"
                        />
                        <Input
                          label={t('enhance.lowlightBs') || 'Saturation weight (bs)'}
                          type="number"
                          value={params.lowlight_bs.toString()}
                          step={0.1}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value)
                            setParams((prev) => ({
                              ...prev,
                              lowlight_bs: isNaN(val) ? 1.0 : val
                            }))
                          }}
                          placeholder="1.0"
                        />
                        <Input
                          label={t('enhance.lowlightBe') || 'Well-exposedness weight (be)'}
                          type="number"
                          value={params.lowlight_be.toString()}
                          step={0.1}
                          onChange={(e) => {
                            const val = parseFloat(e.target.value)
                            setParams((prev) => ({
                              ...prev,
                              lowlight_be: isNaN(val) ? 1.0 : val
                            }))
                          }}
                          placeholder="1.0"
                        />
                      </div>
                    </>
                  )}
                </div>

                {/* Order */}
                {activeMethods.length > 0 && (
                  <div className="parameter-group">
                    <h3 className="parameter-group-title">{t('enhance.order')}</h3>
                    <p className="order-hint">{t('enhance.orderHint')}</p>
                    <div className="order-list">
                      {finalOrder.map((method, index) => (
                        <div
                          key={`${method}-${index}`}
                          className={`order-item ${draggedIndex === index ? 'dragging' : ''}`}
                          draggable
                          onDragStart={() => handleDragStart(index)}
                          onDragOver={(e) => handleDragOverOrder(e, index)}
                          onDragEnd={handleDragEnd}
                        >
                          <span className="order-drag-handle">☰</span>
                          <span className="order-method-name">{getMethodName(method)}</span>
                          <span className="order-number">{index + 1}</span>
                        </div>
                      ))}
                    </div>
                    <button
                      className="order-reset-button"
                      onClick={() => {
                        const newOrder = getActiveMethods()
                        setParams(prev => ({ ...prev, order: newOrder }))
                      }}
                    >
                      {t('enhance.resetOrder')}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <Snackbar
        message={snackbar.message}
        type={snackbar.type}
        isOpen={snackbar.isOpen}
        onClose={() => setSnackbar({ ...snackbar, isOpen: false })}
      />
    </div>
  )
}

export default EnhanceYourImage
