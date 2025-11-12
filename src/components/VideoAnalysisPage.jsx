import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import VideoPlayer from './VideoPlayer'
import TranscriptPanel from './TranscriptPanel'
import CoachingTab from './CoachingTab'
import AnalyticsTab from './AnalyticsTab'
import Chatbot from './Chatbot'
import EnhancementTab from './EnhancementTab'
import './VideoAnalysisPage.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const VideoAnalysisPage = () => {
  const { sessionId } = useParams()
  const navigate = useNavigate()
  const [videoUrl, setVideoUrl] = useState(null)
  const [transcript, setTranscript] = useState([])
  const [acousticFillers, setAcousticFillers] = useState([])
  const [feedback, setFeedback] = useState(null)
  const [reportData, setReportData] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeView, setActiveView] = useState('coaching') // 'coaching', 'analytics', 'chatbot'
  const [showAnalytics, setShowAnalytics] = useState(false)
  const [exportJobId, setExportJobId] = useState(null)
  const [exportStatus, setExportStatus] = useState(null)
  const [exportError, setExportError] = useState(null)
  const exportPollTimeoutRef = useRef(null)

  useEffect(() => {
    if (!sessionId) {
      setError('No session ID provided')
      setLoading(false)
      setAcousticFillers([])
      return
    }

    loadData()
  }, [sessionId])

  const loadData = async () => {
    try {
      setLoading(true)
      setError(null)

      // Load report data
      const reportResponse = await fetch(`${API_BASE_URL}/api/report/${sessionId}`)
      if (!reportResponse.ok) {
        throw new Error('Failed to load report')
      }
      const report = await reportResponse.json()
      setReportData(report)

      const acousticEvents = Array.isArray(report?.filler_analysis?.acoustic_events)
        ? report.filler_analysis.acoustic_events
        : []

      const textModelEvents = Array.isArray(report?.filler_analysis?.text_model_fillers)
        ? report.filler_analysis.text_model_fillers
        : []

      const normalizedTextEvents = textModelEvents
        .filter((event) => typeof event?.start === 'number')
        .map((event) => {
          const start = typeof event?.start === 'number' ? event.start : null
          const end = typeof event?.end === 'number'
            ? event.end
            : (start !== null ? start + 0.3 : null)
          const duration = (start !== null && end !== null)
            ? Math.max(0, end - start)
            : (start !== null ? 0.3 : 0)

          return {
            label: event?.token_original || event?.label || 'filler',
            start,
            end,
            duration,
            confidence: typeof event?.score === 'number'
              ? event.score
              : (typeof event?.confidence === 'number' ? event.confidence : 0),
            method: event?.method || 'text_model'
          }
        })

      const combinedEvents = [
        ...acousticEvents,
        ...normalizedTextEvents
      ].filter((event) => typeof event?.start === 'number')

      setAcousticFillers(combinedEvents)

      // Load video file URL
      const videoFileUrl = `${API_BASE_URL}/api/video/${sessionId}/file`
      setVideoUrl(videoFileUrl)

      // Load timestamped transcript
      const transcriptResponse = await fetch(`${API_BASE_URL}/api/video/${sessionId}/transcript`)
      if (transcriptResponse.ok) {
        const transcriptData = await transcriptResponse.json()
        setTranscript(transcriptData.transcript || [])
      }

      // Load detailed feedback
      const feedbackResponse = await fetch(`${API_BASE_URL}/api/video/${sessionId}/feedback`, {
        method: 'POST'
      })
      if (feedbackResponse.ok) {
        const feedbackData = await feedbackResponse.json()
        setFeedback(feedbackData)
      }

      setLoading(false)
    } catch (err) {
      console.error('Error loading data:', err)
      setError(err.message || 'Failed to load analysis data')
      setLoading(false)
      setAcousticFillers([])
    }
  }

  const handleTimeUpdate = (time) => {
    setCurrentTime(time)
  }

  const handleSeek = (time) => {
    setCurrentTime(time)
  }

  const downloadExportedPdf = async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/api/video/export/${jobId}/download`)
    if (!response.ok) {
      throw new Error(await response.text() || 'Failed to download export')
    }
    const blob = await response.blob()
    if (blob.size === 0) {
      throw new Error('Export file is empty')
    }

    const url = window.URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `Face2Phase_Report_${sessionId}.pdf`
    document.body.appendChild(anchor)
    anchor.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(anchor)
  }

  const handleExportReport = async () => {
    if (!sessionId || exportJobId) return
    try {
      setExportError(null)
      setExportStatus({ status: 'queued', message: 'Preparing export...' })
      const response = await fetch(`${API_BASE_URL}/api/video/${sessionId}/export/pdf`, {
        method: 'POST'
      })
      if (!response.ok) {
        throw new Error(await response.text() || 'Failed to start export')
        }
      const data = await response.json()
      setExportJobId(data.job_id)
      setExportStatus({ status: data.status, message: 'Export queued' })
    } catch (error) {
      console.error('Error starting export:', error)
      setExportError(error.message || 'Failed to start export')
      setExportStatus(null)
    }
  }

  useEffect(() => {
    return () => {
      if (exportPollTimeoutRef.current) {
        clearTimeout(exportPollTimeoutRef.current)
      }
    }
  }, [])

  useEffect(() => {
    if (!exportJobId) return

    let cancelled = false

    const pollStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/video/export/${exportJobId}/status`)
        if (!response.ok) {
          throw new Error(await response.text() || 'Failed to fetch export status')
        }
        const job = await response.json()
        if (cancelled) return

        setExportStatus(job)

        if (job.status === 'completed') {
          await downloadExportedPdf(exportJobId)
          setExportJobId(null)
          setExportStatus({ status: 'completed', message: 'Export downloaded' })
        } else if (job.status === 'failed') {
          setExportError(job.message || 'Export failed')
          setExportJobId(null)
        } else {
          exportPollTimeoutRef.current = setTimeout(pollStatus, 1500)
        }
      } catch (error) {
        if (!cancelled) {
          console.error('Export status error:', error)
          setExportError(error.message || 'Export failed')
          setExportJobId(null)
        }
      }
    }

    pollStatus()

    return () => {
      cancelled = true
      if (exportPollTimeoutRef.current) {
        clearTimeout(exportPollTimeoutRef.current)
    }
  }
  }, [exportJobId])

  if (loading) {
    return (
      <div className="video-analysis-loading">
        <div className="loading-spinner">
          <i className="fas fa-spinner fa-spin" />
        </div>
        <p>Loading analysis...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="video-analysis-error">
        <div className="error-icon">
          <i className="fas fa-exclamation-triangle" />
        </div>
        <h2>Error Loading Analysis</h2>
        <p>{error}</p>
        <button onClick={() => navigate('/dashboard')} className="back-btn">
          Back to Dashboard
        </button>
      </div>
    )
  }

  // Calculate coaching count
  const coachingCount = feedback ? (
    (feedback.strength ? 1 : 0) +
    (feedback.growth_areas?.length > 0 ? 1 : 0) +
    (feedback.follow_up_questions?.length > 0 ? 1 : 0) +
    (feedback.tone ? 1 : 0) +
    (feedback.visual_presence ? 1 : 0) +
    (feedback.conciseness ? 1 : 0) +
    (feedback.summary ? 1 : 0) +
    (feedback.pronunciation ? 1 : 0) +
    (feedback.transcript_improvement?.success ? 1 : 0) +
    (feedback.vocabulary_enhancements?.total_suggestions > 0 ? 1 : 0)
  ) : 0

  return (
    <div className="video-analysis-page">
      <div className="video-analysis-header">
        <button onClick={() => navigate('/dashboard')} className="back-button">
          <i className="fas fa-arrow-left" /> Back
        </button>
        <h1>Video Analysis</h1>
      </div>

      <div className="video-analysis-layout">
        {/* Left Sidebar */}
        <div className="sidebar-navigation">
          <button
            className={`sidebar-btn ${activeView === 'coaching' ? 'active' : ''}`}
            onClick={() => {
              setActiveView('coaching')
              setShowAnalytics(false)
            }}
          >
            <i className="fas fa-comments" />
            <span>Coaching</span>
            {coachingCount > 0 && <span className="badge">{coachingCount}</span>}
          </button>
          <button
            className={`sidebar-btn ${activeView === 'analytics' ? 'active' : ''}`}
            onClick={() => {
              setActiveView('analytics')
              setShowAnalytics(true)
            }}
          >
            <i className="fas fa-chart-bar" />
            <span>Analytics</span>
          </button>
          <button
            className={`sidebar-btn ${activeView === 'chatbot' ? 'active' : ''}`}
            onClick={() => {
              setActiveView('chatbot')
              setShowAnalytics(false)
            }}
          >
            <i className="fas fa-robot" />
            <span>Chatbot</span>
          </button>
          <button
            className={`sidebar-btn ${activeView === 'enhancement' ? 'active' : ''}`}
            onClick={() => {
              setActiveView('enhancement')
              setShowAnalytics(false)
            }}
          >
            <i className="fas fa-magic" />
            <span>Improve</span>
          </button>
          <button
            className={`sidebar-btn ${activeView === 'export' ? 'active' : ''}`}
            onClick={() => {
              handleExportReport()
            }}
            disabled={!!exportJobId}
          >
            <i className="fas fa-download" />
            <span>Export</span>
          </button>
          {(exportStatus || exportError) && (
            <div className="export-status-panel">
              {exportStatus && (
                <div className={`export-status-row status-${exportStatus.status}`}>
                  {exportStatus.status === 'completed' ? (
                    <i className="fas fa-check-circle" />
                  ) : exportStatus.status === 'failed' ? (
                    <i className="fas fa-times-circle" />
                  ) : (
                    <i className="fas fa-spinner fa-spin" />
                  )}
                  <span>{exportStatus.message || `Status: ${exportStatus.status}`}</span>
                </div>
              )}
              {exportError && (
                <div className="export-status-row status-failed">
                  <i className="fas fa-exclamation-triangle" />
                  <span>{exportError}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Middle: Video and Transcript */}
        <div className="video-content-area">
          <motion.div
            className="video-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            {videoUrl && (
              <VideoPlayer
                videoUrl={videoUrl}
                onTimeUpdate={handleTimeUpdate}
                currentTime={currentTime}
              />
            )}
          </motion.div>

          <motion.div
            className="transcript-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <TranscriptPanel
              transcript={transcript}
              currentTime={currentTime}
              onSeek={handleSeek}
              acousticFillers={acousticFillers}
            />
          </motion.div>

          {/* Analytics Button */}
          <motion.div
            className="analytics-button-container"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            style={{ display: 'flex', justifyContent: 'center', padding: '0.5rem 0' }}
          >
            <button
              className={`analytics-toggle-btn ${showAnalytics ? 'active' : ''}`}
              onClick={() => {
                const newShowAnalytics = !showAnalytics
                setShowAnalytics(newShowAnalytics)
                if (newShowAnalytics) {
                  setActiveView('analytics')
                } else if (activeView === 'analytics') {
                  setActiveView('coaching')
                }
              }}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.5rem 1rem',
                background: showAnalytics ? 'var(--accent-primary)' : 'var(--bg-secondary)',
                border: '1px solid var(--border-color)',
                color: showAnalytics ? 'white' : 'var(--text-primary)',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.75rem',
                transition: 'all 0.2s'
              }}
            >
              <i className={`fas fa-chevron-${showAnalytics ? 'down' : 'up'}`} />
              <span>{showAnalytics ? 'Hide' : 'Show'} Analytics</span>
            </button>
          </motion.div>
        </div>

        {/* Right Content Area */}
        <div className="content-panel">
          {activeView === 'coaching' && (
            <motion.div
              className="content-section"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              key="coaching"
            >
              <CoachingTab feedback={feedback} reportData={reportData} />
            </motion.div>
          )}

          {activeView === 'analytics' && (
            <motion.div
              className="content-section"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              key="analytics"
            >
              <AnalyticsTab reportData={reportData} />
            </motion.div>
          )}

          {activeView === 'chatbot' && (
            <motion.div
              className="content-section chatbot-section"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              key="chatbot"
            >
              <Chatbot sessionId={sessionId} />
            </motion.div>
          )}

          {activeView === 'enhancement' && (
            <motion.div
              className="content-section"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              key="enhancement"
            >
              <EnhancementTab sessionId={sessionId} reportData={reportData} />
            </motion.div>
          )}
        </div>
      </div>
    </div>
  )
}

export default VideoAnalysisPage
