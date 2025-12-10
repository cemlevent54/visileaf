import { useRef, useState, useMemo } from 'react'
import Navbar from '../../components/navbar/Navbar'
import Snackbar from '../../components/snackbar/Snackbar'
import enhancementService from '../../services/enhancement.service'
import { useTranslation } from '../../hooks/useTranslation'
import Select from '../../components/select/Select'
import './EnhanceYourImageWithDL.css'

function EnhanceYourImageWithDL() {
  const { t } = useTranslation()
  
  // Model seçeneklerini i18n'den çek
  const MODEL_OPTIONS = useMemo(() => [
    { value: 'enlightengan', label: t('enhanceDL.models.enlightengan') },
    { value: 'zero_dce', label: t('enhanceDL.models.zero_dce') },
    { value: 'llflow', label: t('enhanceDL.models.llflow') },
    { value: 'mirnet_v2', label: t('enhanceDL.models.mirnet_v2') },
  ], [t])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [inputImage, setInputImage] = useState<File | null>(null)
  const [inputImagePreview, setInputImagePreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [modelName, setModelName] = useState<string>('')
  const [outputImage, setOutputImage] = useState<string | null>(null)

  const [snackbar, setSnackbar] = useState<{
    isOpen: boolean
    message: string
    type: 'success' | 'error' | 'info' | 'warning'
  }>({
    isOpen: false,
    message: '',
    type: 'info'
  })

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

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
      reader.onloadend = () => setInputImagePreview(reader.result as string)
      reader.readAsDataURL(file)
      setOutputImage(null)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
    const file = e.dataTransfer.files[0]
      const changeEvent = {
        target: { files: [file] }
      } as unknown as React.ChangeEvent<HTMLInputElement>
      handleImageSelect(changeEvent)
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

    if (!modelName) {
      setSnackbar({
        isOpen: true,
        message: t('enhanceDL.modelRequired'),
        type: 'warning'
      })
      return
    }

    setLoading(true)
    setSnackbar({ isOpen: false, message: '', type: 'info' })

    try {
      const response = await enhancementService.enhanceImageWithDeepLearning(
        inputImage,
        modelName
      )
      
      const imageUrl = URL.createObjectURL(response)
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

  const handleClear = () => {
    setInputImage(null)
    setInputImagePreview(null)
    setOutputImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
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

  return (
    <div className="enhance-page">
      <Navbar />
      <main className="enhance-main">
        <div className="enhance-container">
          <h1 className="enhance-title">
            {t('enhanceDL.title')}
          </h1>
          <p className="enhance-subtitle">
            {t('enhanceDL.subtitle')}
          </p>

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

              {/* Actions placed above Output */}
              <div className="action-section">
                <button
                  className="enhance-button"
                  onClick={handleEnhance}
                  disabled={!inputImage || !modelName || loading}
                >
                  {loading ? t('enhance.processing') : t('enhance.enhanceButton')}
                </button>
                <button className="clear-secondary" onClick={handleClear}>
                  {t('enhanceDL.clear')}
                </button>
              </div>

              {/* Output Section */}
              <div className="image-section">
                <h2 className="section-title">{t('enhanceDL.outputTitle')}</h2>
                <div className="response-area">
                  {outputImage ? (
                    <div className="image-preview-container">
                      <img
                        src={outputImage}
                        alt="DL output"
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
                      <p className="placeholder-text">
                        {t('enhanceDL.outputPlaceholder')}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Model Selection */}
            <div className="enhance-parameters-column">
              <h2 className="parameters-title">{t('enhanceDL.modelSelectionTitle')}</h2>
              <div className="parameters-content">
                <Select
                  label={t('enhanceDL.modelLabel')}
                  value={modelName}
                  onChange={setModelName}
                  options={MODEL_OPTIONS}
                  placeholder={t('enhanceDL.modelPlaceholder')}
                  hint={t('enhanceDL.modelHint')}
                />
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

export default EnhanceYourImageWithDL

