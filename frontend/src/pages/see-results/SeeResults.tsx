import { useEffect, useState } from 'react'
import Navbar from '../../components/navbar/Navbar'
import { useTranslation } from '../../hooks/useTranslation'
import enhancementService from '../../services/enhancement.service'
import { getApiUrl } from '../../services/api.interceptor'
import './SeeResults.css'

interface EnhancementResultInput {
  id: string | null
  path: string | null
  width: number | null
  height: number | null
  created_at: string | null
}

interface EnhancementResult {
  id: string
  enhancement_type: string | null
  created_at: string
  output_path: string
  output_width: number | null
  output_height: number | null
  params: Record<string, any>
  input: EnhancementResultInput
}

function SeeResults() {
  const { t } = useTranslation()
  const [results, setResults] = useState<EnhancementResult[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selectedResult, setSelectedResult] = useState<EnhancementResult | null>(null)
  const [modalType, setModalType] = useState<'images' | 'params' | 'all' | null>(null)

  const apiBaseUrl = getApiUrl()

  const buildImageUrl = (path?: string | null): string => {
    if (!path) return ''
    const base = apiBaseUrl.replace(/\/$/, '')
    const cleanPath = path.replace(/^\/+/, '')
    return `${base}/${cleanPath}`
  }

  useEffect(() => {
    const fetchResults = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await enhancementService.listResults()
        setResults(data)
      } catch (e: any) {
        setError(e?.message || 'Failed to load results')
      } finally {
        setLoading(false)
      }
    }

    fetchResults()
  }, [])

  const handleActionClick = (result: EnhancementResult, type: 'images' | 'params' | 'all' | 'export') => {
    if (type === 'export') {
      // Şimdilik mock: sadece console.log yap
      // İleride gerçek export (JSON / CSV / ZIP) için buraya gelecek
      // eslint-disable-next-line no-console
      console.log('Export clicked for result', result)
      return
    }
    setSelectedResult(result)
    setModalType(type)
  }

  const closeModal = () => {
    setSelectedResult(null)
    setModalType(null)
  }

  return (
    <div className="app">
      <Navbar />
      <main className="see-results-page">
        <h1 className="see-results-title">{t('results.title') || 'See Results'}</h1>

        {loading && <p className="see-results-status">{t('results.loading') || 'Loading results...'}</p>}
        {error && <p className="see-results-error">{error}</p>}

        {!loading && !error && results.length === 0 && (
          <p className="see-results-status">{t('results.empty') || 'No enhancement results yet.'}</p>
        )}

        {!loading && !error && results.length > 0 && (
          <div className="see-results-table-wrapper">
            <table className="see-results-table">
              <thead>
                <tr>
                  <th>{t('results.table.id') || 'ID'}</th>
                  <th>{t('results.table.input') || 'Input'}</th>
                  <th>{t('results.table.output') || 'Output'}</th>
                  <th>{t('results.table.actions') || 'Actions'}</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => {
                  const inputUrl = buildImageUrl(r.input?.path)
                  const outputUrl = buildImageUrl(r.output_path)
                  return (
                    <tr key={r.id}>
                      <td className="see-results-id">{r.id}</td>
                      <td>
                        {inputUrl ? (
                          <img
                            src={inputUrl}
                            alt="Input"
                            className="see-results-thumb"
                          />
                        ) : (
                          <span>-</span>
                        )}
                      </td>
                      <td>
                        {outputUrl ? (
                          <img
                            src={outputUrl}
                            alt="Output"
                            className="see-results-thumb"
                          />
                        ) : (
                          <span>-</span>
                        )}
                      </td>
                      <td>
                        <div className="see-results-actions">
                          <button
                            type="button"
                            className="see-results-action-button"
                            onClick={() => handleActionClick(r, 'images')}
                          >
                            {t('results.actions.showImages') || 'Show input-output images'}
                          </button>
                          <button
                            type="button"
                            className="see-results-action-button"
                            onClick={() => handleActionClick(r, 'params')}
                          >
                            {t('results.actions.showParams') || 'Show parameters'}
                          </button>
                          <button
                            type="button"
                            className="see-results-action-button"
                            onClick={() => handleActionClick(r, 'all')}
                          >
                            {t('results.actions.showAll') || 'Show everything'}
                          </button>
                          <button
                            type="button"
                            className="see-results-action-button secondary"
                            onClick={() => handleActionClick(r, 'export')}
                          >
                            {t('results.actions.export') || 'Export'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        {selectedResult && modalType && (
          <div className="see-results-modal-backdrop" onClick={closeModal}>
            <div className="see-results-modal" onClick={(e) => e.stopPropagation()}>
              <div className="see-results-modal-header">
                <h2>
                  {modalType === 'images'
                    ? t('results.modal.imagesTitle') || 'Input / Output Images'
                    : modalType === 'params'
                      ? t('results.modal.paramsTitle') || 'Parameters'
                      : t('results.modal.allTitle') || 'Details'}
                </h2>
                <button type="button" className="see-results-close" onClick={closeModal}>
                  ×
                </button>
              </div>

              <div className="see-results-modal-content">
                {(modalType === 'images' || modalType === 'all') && (
                  <div className="see-results-images-preview">
                    <div>
                      <h3>{t('results.modal.input') || 'Input'}</h3>
                      {(() => {
                        const inputUrl = buildImageUrl(selectedResult.input?.path)
                        return inputUrl ? (
                          <>
                            <img src={inputUrl} alt="Input" className="see-results-img" />
                            <button
                              type="button"
                              className="see-results-image-button"
                              onClick={() => window.open(inputUrl, '_blank', 'noopener,noreferrer')}
                            >
                              {t('results.actions.viewInputImage') || 'View input image'}
                            </button>
                          </>
                        ) : (
                          <p>-</p>
                        )
                      })()}
                    </div>
                    <div>
                      <h3>{t('results.modal.output') || 'Output'}</h3>
                      {(() => {
                        const outputUrl = buildImageUrl(selectedResult.output_path)
                        return outputUrl ? (
                          <>
                            <img src={outputUrl} alt="Output" className="see-results-img" />
                            <button
                              type="button"
                              className="see-results-image-button"
                              onClick={() => window.open(outputUrl, '_blank', 'noopener,noreferrer')}
                            >
                              {t('results.actions.viewOutputImage') || 'View output image'}
                            </button>
                          </>
                        ) : (
                          <p>-</p>
                        )
                      })()}
                    </div>
                  </div>
                )}

                {(modalType === 'params' || modalType === 'all') && (
                  <div className="see-results-params">
                    <h3>{t('results.modal.params') || 'Parameters'}</h3>
                    {Object.keys(selectedResult.params || {}).length === 0 ? (
                      <p className="see-results-status">
                        {t('results.noParams') || 'No parameters'}
                      </p>
                    ) : (
                      <ul className="see-results-params-list">
                        {Object.entries(selectedResult.params).map(([key, value]) => (
                          <li key={key}>
                            <strong>{key}:</strong>{' '}
                            <span>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default SeeResults


