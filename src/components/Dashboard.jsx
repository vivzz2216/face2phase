import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../context/AuthContext'
import { API_BASE_URL, toAbsoluteUrl } from '../lib/url'
import LoadingScreen from './LoadingScreen'
import './Dashboard.css'
const GUEST_HISTORY_STORAGE_KEY = 'face2phase_guest_history_v1'

const deriveDefaultProjectName = (filename) => {
  if (!filename || typeof filename !== 'string') return ''
  const withoutExtension = filename.replace(/\.[^/.]+$/, '')
  return withoutExtension.replace(/[_\-]+/g, ' ').replace(/\s+/g, ' ').trim()
}

const Dashboard = () => {
  const navigate = useNavigate()
  const { user, loading, signInWithGoogle, logout } = useAuth()
  const [showLogin, setShowLogin] = useState(false)
  const [file, setFile] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [projectName, setProjectName] = useState('')
  const [projectNameDraft, setProjectNameDraft] = useState('')
  const [projectNameError, setProjectNameError] = useState('')
  const [showProjectModal, setShowProjectModal] = useState(false)
  const [analysisHistory, setAnalysisHistory] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [retentionFilter, setRetentionFilter] = useState('recent')
  const [selectedSessions, setSelectedSessions] = useState([])
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const fileInputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const recordedChunksRef = useRef([])
  const recordingTimerRef = useRef(null)
  const componentActiveRef = useRef(true)
  const discardRecordingRef = useRef(false)
  const previewVideoRef = useRef(null)
  const isBrowser = typeof window !== 'undefined'
  const [isGuest, setIsGuest] = useState(!user)
  const [supportsRecording, setSupportsRecording] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingType, setRecordingType] = useState(null)
  const [recordingError, setRecordingError] = useState('')
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [previewStream, setPreviewStream] = useState(null)

  useEffect(() => {
    setIsGuest(!user)
  }, [user])

  useEffect(() => {
    componentActiveRef.current = true
    if (typeof window !== 'undefined' && navigator?.mediaDevices && typeof window.MediaRecorder !== 'undefined') {
      setSupportsRecording(true)
    }
    return () => {
      componentActiveRef.current = false
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      setPreviewStream(null)
    }
  }, [])

  useEffect(() => {
    const videoEl = previewVideoRef.current
    if (!videoEl) return
    if (previewStream) {
      videoEl.srcObject = previewStream
      videoEl.play?.().catch(() => { })
    } else {
      videoEl.srcObject = null
    }
  }, [previewStream])

  const normalizeSummary = useCallback((session) => {
    if (!session) return null
    const breakdown = session.score_breakdown || {}
    const metrics = session.metrics || {}
    const highlightsRaw = session.highlights
    const highlights = (highlightsRaw && typeof highlightsRaw === 'object') ? highlightsRaw : {}
    const thumbnailCandidate =
      session.thumbnail_url ||
      highlights.thumbnail_url ||
      session.thumbnail ||
      null
    const speakingMetrics = session.speaking_metrics || metrics.speaking_metrics || {}
    const durationSeconds =
      metrics.duration_seconds ??
      metrics.total_duration ??
      metrics.duration ??
      metrics.session_length ??
      speakingMetrics.total_duration ??
      session.duration_seconds ??
      session.duration ??
      null
    const pacingScore =
      metrics.pacing_score ??
      metrics.pacing ??
      metrics.voice_pacing_score ??
      metrics.fluency_score ??
      metrics.speaking_rate_wpm ??
      speakingMetrics.speaking_rate_wpm ??
      session.pacing_value ??
      null
    const totalWords =
      metrics.total_words ??
      speakingMetrics.total_words ??
      session.total_words ??
      (session.word_analysis?.vocabulary?.total_words) ??
      null
    let fillerPct =
      metrics.filler_ratio ??
      (session.filler_word_ratio !== undefined ? session.filler_word_ratio * 100 : undefined) ??
      (session.metrics?.filler_word_ratio !== undefined ? session.metrics.filler_word_ratio * 100 : undefined) ??
      (metrics.filler_percentage !== undefined ? metrics.filler_percentage : undefined)
    if (typeof fillerPct === 'number' && fillerPct <= 1) {
      fillerPct *= 100
    }

    const originalFileName =
      session.original_file_name ||
      session.source_file_name ||
      session.file_name ||
      session.filename ||
      (metrics.original_file_name || null)
    const projectLabel =
      session.project_name ||
      session.title ||
      session.display_title ||
      metrics.project_name ||
      null
    const displayTitle =
      projectLabel ||
      deriveDefaultProjectName(originalFileName || '') ||
      session.file_name ||
      'Untitled Session'

    return {
      ...session,
      file_name: displayTitle,
      display_title: displayTitle,
      project_name: projectLabel || displayTitle,
      original_file_name: originalFileName,
      overall_score: session.overall_score ?? breakdown.overall_score ?? null,
      voice_confidence: session.voice_confidence ?? breakdown.voice_confidence ?? session.metrics?.voice_confidence ?? null,
      facial_confidence: session.facial_confidence ?? breakdown.facial_confidence ?? session.metrics?.facial_confidence ?? null,
      vocabulary_score: session.vocabulary_score ?? breakdown.vocabulary_score ?? session.metrics?.vocabulary_score ?? null,
      created_at: session.created_at || session.timestamp || session.updated_at || null,
      duration_seconds: durationSeconds,
      pacing_value: pacingScore,
      total_time_label: session.total_time || session.duration_label || null,
      total_words: totalWords,
      filler_percentage: typeof fillerPct === 'number' ? fillerPct : null,
      thumbnail_url: toAbsoluteUrl(thumbnailCandidate)
    }
  }, [toAbsoluteUrl])

  const getStoredGuestHistory = useCallback(() => {
    if (!isBrowser) return []
    try {
      const raw = window.localStorage.getItem(GUEST_HISTORY_STORAGE_KEY)
      if (!raw) return []
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : []
    } catch (error) {
      console.warn('Failed to read guest history from storage', error)
      return []
    }
  }, [isBrowser])

  const writeGuestHistory = useCallback((entries) => {
    if (!isBrowser) return
    try {
      window.localStorage.setItem(GUEST_HISTORY_STORAGE_KEY, JSON.stringify(entries))
    } catch (error) {
      console.warn('Failed to persist guest history', error)
    }
  }, [isBrowser])

  const normalizeLegacyAnalysis = useCallback((analysis) => {
    if (!analysis) return null
    const audio = analysis.audio_analysis || {}
    const facial = analysis.facial_analysis || {}
    const text = analysis.text_analysis || {}
    const metrics = analysis.metrics || {}
    const speakingMetrics = audio.speaking_metrics || analysis.speaking_metrics || {}
    const durationSeconds =
      metrics.duration_seconds ??
      metrics.total_duration ??
      audio.duration_seconds ??
      audio.total_duration ??
      speakingMetrics.total_duration ??
      analysis.duration_seconds ??
      analysis.duration ??
      null
    const pacingScore =
      metrics.pacing_score ??
      audio.pacing_score ??
      audio.fluency_score ??
      audio.voice_confidence_score ??
      audio.speaking_rate_wpm ??
      speakingMetrics.speaking_rate_wpm ??
      null
    const totalWords =
      metrics.total_words ??
      speakingMetrics.total_words ??
      audio.total_words ??
      (text.vocabulary_metrics?.total_words) ??
      null
    let fillerPct =
      metrics.filler_ratio ??
      audio.filler_analysis?.filler_ratio ??
      (analysis.filler_word_ratio !== undefined ? analysis.filler_word_ratio * 100 : undefined)
    if (typeof fillerPct === 'number' && fillerPct <= 1) {
      fillerPct *= 100
    }

    return {
      session_id: analysis.session_id || analysis.id,
      file_name: analysis.file_name || 'Untitled Session',
      file_type: analysis.file_type || 'audio',
      overall_score: analysis.overall_score ?? audio.overall_score ?? null,
      voice_confidence: analysis.voice_confidence ?? audio.voice_confidence_score ?? null,
      facial_confidence: analysis.facial_confidence ?? facial.facial_confidence_score ?? null,
      vocabulary_score: analysis.vocabulary_score ?? text.vocabulary_score ?? null,
      created_at: analysis.created_at || analysis.timestamp || null,
      duration_seconds: durationSeconds,
      pacing_value: pacingScore,
      total_time_label: analysis.total_time || audio.total_time || null,
      total_words: totalWords,
      filler_percentage: typeof fillerPct === 'number' ? fillerPct : null,
      thumbnail_url: toAbsoluteUrl(analysis.thumbnail_url || analysis.thumbnail || null)
    }
  }, [toAbsoluteUrl])

  const syncGuestHistoryFromStorage = useCallback(() => {
    // FIX 5: Only sync if user is not logged in (defensive check)
    if (user) {
      setAnalysisHistory([])
      setLoadingHistory(false)
      return
    }

    const stored = getStoredGuestHistory()
    if (!stored.length) {
      setAnalysisHistory([])
      setLoadingHistory(false)
      return
    }
    const normalized = stored
      .map((entry) => normalizeSummary(entry))
      .filter(Boolean)
    setAnalysisHistory(normalized)
    setLoadingHistory(false)
  }, [getStoredGuestHistory, normalizeSummary, user])

  const persistGuestSession = useCallback((session) => {
    if (!session?.session_id) return
    const normalized = normalizeSummary(session)
    if (!normalized) return

    const existing = getStoredGuestHistory()
    const existingIndex = existing.findIndex((item) => item.session_id === normalized.session_id)

    let updated
    if (existingIndex >= 0) {
      const copy = [...existing]
      copy[existingIndex] = { ...copy[existingIndex], ...normalized }
      updated = copy
    } else {
      updated = [normalized, ...existing]
    }

    writeGuestHistory(updated)
    setAnalysisHistory(updated)
    setLoadingHistory(false)
  }, [getStoredGuestHistory, normalizeSummary, writeGuestHistory])

  const openProjectModal = useCallback(() => {
    const fallbackName =
      (projectName && projectName.trim()) ||
      deriveDefaultProjectName(file?.name) ||
      ''
    setProjectNameDraft(fallbackName)
    setProjectNameError('')
    setShowProjectModal(true)
  }, [file, projectName])

  const closeProjectModal = useCallback(() => {
    setShowProjectModal(false)
    setProjectNameError('')
  }, [])

  // Update showLogin based on user state
  useEffect(() => {
    if (loading) return
    setShowLogin(!user)
  }, [user, loading])

  // Handle file passed from navbar
  useEffect(() => {
    const state = window.history.state
    if (state && state.usr && state.usr.fileToUpload) {
      setFile(state.usr.fileToUpload)
      // Clear state
      window.history.replaceState({ ...state, usr: null }, '')
    }
  }, [])

  const loadAnalysisHistory = useCallback(async () => {
    if (!user) {
      setLoadingHistory(true)
      syncGuestHistoryFromStorage()
      return []
    }

    // FIX 1: Clear any guest history first before loading logged-in user data
    setAnalysisHistory([])
    setLoadingHistory(true)

    const normalizedSessions = []
    try {
      const sessionsRequestPayload = {
        email: user.email,
        uid: user.uid,
        display_name: user.displayName || null
      }

      try {
        console.debug('[History] Requesting Firebase sessions', sessionsRequestPayload)
        const sessionResponse = await fetch(`${API_BASE_URL}/api/firebase/sessions`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            ...sessionsRequestPayload,
            limit: 100
          })
        })

        if (sessionResponse.ok) {
          const data = await sessionResponse.json()
          console.debug('[History] Firebase sessions response', data)
          console.log('[History] ✓ Successfully fetched sessions:', data.sessions?.length || 0, 'sessions')
          const sessionList = Array.isArray(data.sessions) ? data.sessions : []
          sessionList.forEach((session) => {
            const normalized = normalizeSummary(session)
            if (normalized) normalizedSessions.push(normalized)
          })
        } else {
          // FIX 3: Improved error handling
          const detailText = await sessionResponse.text().catch(() => '')
          console.error('[History] ✗ Firebase sessions request failed:', sessionResponse.status, detailText)
          console.warn('Session summaries request failed:', sessionResponse.status, detailText)
        }
      } catch (sessionError) {
        // FIX 3: Improved error handling
        console.error('[History] ✗ Firebase sessions request error (network or API issue):', sessionError)
        console.warn('Session summary endpoint unavailable, falling back to legacy analyses.', sessionError)
      }

      if (normalizedSessions.length === 0) {
        console.log('[History] No sessions found in primary endpoint, trying legacy analyses...')
        try {
          console.debug('[History] Requesting legacy analyses', sessionsRequestPayload)
          const legacyResponse = await fetch(`${API_BASE_URL}/api/firebase/analyses?limit=1000`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(sessionsRequestPayload)
          })

          if (legacyResponse.ok) {
            const legacyData = await legacyResponse.json()
            console.debug('[History] Legacy analyses response', legacyData)
            console.log('[History] ✓ Successfully fetched legacy analyses:', legacyData.analyses?.length || 0, 'analyses')
            const legacyList = Array.isArray(legacyData.analyses) ? legacyData.analyses : []
            legacyList.forEach((analysis) => {
              const normalizedLegacy = normalizeLegacyAnalysis(analysis)
              if (normalizedLegacy) normalizedSessions.push(normalizedLegacy)
            })
          } else {
            // FIX 3: Improved error handling
            const detailText = await legacyResponse.text().catch(() => '')
            console.error('[History] ✗ Legacy analyses request failed:', legacyResponse.status, detailText)
            console.warn('Legacy analyses request failed:', legacyResponse.status, detailText)
          }
        } catch (legacyError) {
          // FIX 3: Improved error handling
          console.error('[History] ✗ Legacy analyses request error (network or API issue):', legacyError)
          console.warn('Legacy analyses endpoint unavailable.', legacyError)
        }
      }

      const parseDate = (value) => {
        if (!value) return 0
        const timestamp = Date.parse(value)
        return Number.isNaN(timestamp) ? 0 : timestamp
      }

      normalizedSessions.sort((a, b) => parseDate(b.created_at) - parseDate(a.created_at))

      console.log('[History] 📊 Final result: Setting', normalizedSessions.length, 'sessions to state')
      setAnalysisHistory(normalizedSessions)
      setSelectedSessions((prev) => {
        if (prev.length === 0) return prev
        const validIds = new Set(normalizedSessions.map((session) => session.session_id))
        return prev.filter((session) => validIds.has(session.session_id))
      })
      return normalizedSessions
    } catch (error) {
      console.error('[History] Unexpected error loading analysis history:', error)
      return null
    } finally {
      setLoadingHistory(false)
    }
  }, [normalizeLegacyAnalysis, normalizeSummary, syncGuestHistoryFromStorage, toAbsoluteUrl, user])

  // FIX 4: Clear guest history when user logs in
  useEffect(() => {
    if (user && !loading) {
      // User just logged in - clear any guest history
      const guestHistory = getStoredGuestHistory()
      if (guestHistory.length > 0) {
        // Clear guest history from state immediately
        setAnalysisHistory([])
      }
    }
  }, [user, loading, getStoredGuestHistory])

  // Load analysis history only for logged-in users
  // Reload analysis history when loading finishes OR user changes.
  // Include `user` explicitly so history refreshes after sign-in (popup or redirect).
  useEffect(() => {
    if (loading) return
    loadAnalysisHistory()
  }, [loading, user, loadAnalysisHistory])

  const handleFileChange = (e) => {
    if (isRecording) {
      discardRecordingRef.current = true
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
    }
    setPreviewStream(null)
    const selectedFile = e.target.files[0]
    setFile(selectedFile)
    if (selectedFile) {
      const defaultName = deriveDefaultProjectName(selectedFile.name)
      setProjectName(defaultName)
      setProjectNameDraft(defaultName)
      setProjectNameError('')
    } else {
      setProjectName('')
      setProjectNameDraft('')
      setProjectNameError('')
    }
  }

  const pickMimeType = (type) => {
    if (typeof window === 'undefined' || !window.MediaRecorder) return undefined
    const candidates = type === 'video'
      ? [
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=vp9,opus',
        'video/webm'
      ]
      : ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
    return candidates.find((candidate) => window.MediaRecorder.isTypeSupported(candidate))
  }

  const cleanupRecordingState = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
    setIsRecording(false)
    setRecordingType(null)
    setRecordingDuration(0)
    discardRecordingRef.current = false
    setPreviewStream(null)
  }

  const stopStreamTracks = (stream) => {
    if (stream) {
      stream.getTracks().forEach((track) => {
        try {
          track.stop()
        } catch (err) {
          // ignore
        }
      })
    }
  }

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }, [])

  const cancelRecording = () => {
    if (!isRecording) return
    discardRecordingRef.current = true
    stopRecording()
  }

  const startRecording = async (type) => {
    if (!supportsRecording || isRecording || processing) return
    if (!navigator?.mediaDevices?.getUserMedia) {
      setRecordingError('Media recording is not supported in this browser.')
      return
    }

    // FIX #12: Show pre-recording instructions
    const proceedWithRecording = window.confirm(
      `📹 Recording Tips for Best Results:\n\n` +
      `✓ Lighting: Face a window or light source (avoid backlighting)\n` +
      `✓ Audio: Use a quiet room and speak clearly\n` +
      `✓ Camera: Position camera at eye level, 2-3 feet away\n` +
      `✓ Background: Use a clean, uncluttered background\n` +
      `✓ Posture: Sit upright and maintain good eye contact with camera\n\n` +
      `Ready to start recording?`
    )

    if (!proceedWithRecording) {
      return
    }

    try {
      setRecordingError('')
      setFile(null)
      setProjectName('')
      setProjectNameDraft('')
      setProjectNameError('')
      setShowProjectModal(false)
      const videoConstraints = type === 'video'
        ? { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
        : false
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
        video: videoConstraints
      })

      if (type === 'video' && stream.getAudioTracks().length === 0) {
        try {
          const audioFallback = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true }
          })
          audioFallback.getAudioTracks().forEach((track) => stream.addTrack(track))
        } catch (audioError) {
          setRecordingError('Microphone access unavailable; video audio may be missing.')
        }
      }

      const mimeType = pickMimeType(type)
      const recorderOptions = {}
      if (mimeType) {
        recorderOptions.mimeType = mimeType
      }
      if (type === 'video') {
        recorderOptions.audioBitsPerSecond = 128000
        recorderOptions.videoBitsPerSecond = 2_500_000
      } else {
        recorderOptions.audioBitsPerSecond = 128000
      }

      let recorder
      try {
        recorder = Object.keys(recorderOptions).length
          ? new MediaRecorder(stream, recorderOptions)
          : new MediaRecorder(stream)
      } catch (optionError) {
        recorder = new MediaRecorder(stream)
      }
      mediaRecorderRef.current = recorder
      recordedChunksRef.current = []
      discardRecordingRef.current = false
      setRecordingType(type)
      setIsRecording(true)
      setRecordingDuration(0)
      if (type === 'video') {
        const videoOnlyStream = new MediaStream(stream.getVideoTracks())
        setPreviewStream(videoOnlyStream)
      } else {
        setPreviewStream(null)
      }
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1)
      }, 1000)

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        stopStreamTracks(stream)
        const isActive = componentActiveRef.current
        const shouldDiscard = discardRecordingRef.current
        const chunks = recordedChunksRef.current.slice()
        recordedChunksRef.current = []
        cleanupRecordingState()
        if (!isActive || shouldDiscard || chunks.length === 0) {
          return
        }

        const blobType = recorder.mimeType || mimeType || (type === 'video' ? 'video/webm' : 'audio/webm')
        const blob = new Blob(chunks, { type: blobType })
        const extension = blobType.includes('video') ? 'webm' : 'webm'
        const recordedFile = new File(
          [blob],
          `face2phase-${type}-recording-${Date.now()}.${extension}`,
          { type: blob.type }
        )
        if (!componentActiveRef.current) {
          return
        }
        setFile(recordedFile)
        const defaultName = deriveDefaultProjectName(recordedFile.name)
        setProjectName(defaultName)
        setProjectNameDraft(defaultName)
        setProjectNameError('')
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        uploadFile(defaultName, recordedFile)
      }

      recorder.onerror = (event) => {
        stopStreamTracks(stream)
        cleanupRecordingState()
        if (componentActiveRef.current) {
          setRecordingError(event.error?.message || 'Failed to record media.')
        }
      }

      recorder.start()
    } catch (error) {
      cleanupRecordingState()
      setRecordingError(error?.message || 'Unable to access microphone/camera.')
    }
  }

  const formatRecordingTime = (seconds) => {
    const mins = Math.floor(seconds / 60).toString().padStart(2, '0')
    const secs = Math.floor(seconds % 60).toString().padStart(2, '0')
    return `${mins}:${secs}`
  }

  const uploadFile = async (nameOverride, fileOverride = null) => {
    const uploadSource = fileOverride ?? file
    if (!uploadSource) {
      alert('Please select a file first')
      return
    }

    const projectLabel = ((nameOverride ?? projectName ?? deriveDefaultProjectName(uploadSource?.name || '')) || '').trim()
    if (!projectLabel) {
      openProjectModal()
      return
    }

    setProjectName(projectLabel)
    setProjectNameDraft(projectLabel)
    setProcessing(true)
    setProgress(0)

    try {
      const formData = new FormData()
      formData.append('file', uploadSource)
      if (projectLabel) {
        formData.append('project_name', projectLabel)
      }

      // Add user info if logged in
      if (user) {
        formData.append('email', user.email || '')
        formData.append('uid', user.uid || '')
        formData.append('display_name', user.displayName || '')
      }

      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!uploadResponse.ok) {
        let errorMessage = 'Upload failed'
        try {
          const errorData = await uploadResponse.json()
          errorMessage = errorData.detail || errorData.message || errorMessage
        } catch (e) {
          errorMessage = `Server error: ${uploadResponse.status} ${uploadResponse.statusText}`
        }
        throw new Error(errorMessage)
      }

      const uploadData = await uploadResponse.json()

      if (!uploadData.session_id) {
        throw new Error('Server did not return a session ID')
      }

      setSessionId(uploadData.session_id)
      setProgress(20)

      const progressInterval = setInterval(async () => {
        try {
          const progressResponse = await fetch(`${API_BASE_URL}/status/${uploadData.session_id}`)

          if (!progressResponse.ok) {
            if (progressResponse.status === 404) {
              clearInterval(progressInterval)
              setProcessing(false)
              alert('Session not found. Please try uploading again.')
              return
            }
            throw new Error(`Status check failed: ${progressResponse.status}`)
          }

          const progressData = await progressResponse.json()

          setProgress(progressData.progress || 0)

          if (progressData.status === 'error') {
            clearInterval(progressInterval)
            setProcessing(false)
            alert(`Processing failed: ${progressData.error || 'Unknown error'}`)
            return
          }

          if (progressData.progress >= 100 || progressData.status === 'completed') {
            clearInterval(progressInterval)
            setProcessing(false)

            const finalReport = progressData.report || {}
            const sessionStub = {
              session_id: uploadData.session_id,
              project_name: projectLabel,
              title: projectLabel,
              file_name: projectLabel || file?.name,
              original_file_name: file?.name || null,
              file_type:
                finalReport.file_type ||
                uploadData.file_type ||
                (file?.type?.startsWith('video') ? 'video' : file?.type?.startsWith('audio') ? 'audio' : null),
              overall_score: finalReport.overall_score,
              score_breakdown: {
                voice_confidence: finalReport.voice_confidence,
                facial_confidence: finalReport.facial_confidence,
                vocabulary_score: finalReport.vocabulary_score
              },
              metrics: {
                duration_seconds:
                  finalReport.speaking_metrics?.total_duration ??
                  finalReport.total_duration ??
                  finalReport.metrics?.duration_seconds ??
                  null,
                total_duration:
                  finalReport.speaking_metrics?.total_duration ??
                  finalReport.total_duration ??
                  finalReport.metrics?.total_duration ??
                  null,
                pacing_score: finalReport.speaking_rate_wpm,
                speaking_rate_wpm: finalReport.speaking_rate_wpm,
                total_words:
                  finalReport.total_words ??
                  finalReport.speaking_metrics?.total_words ??
                  finalReport.metrics?.total_words ??
                  null,
                filler_ratio: finalReport.filler_word_ratio,
                filler_percentage:
                  typeof finalReport.filler_word_ratio === 'number'
                    ? finalReport.filler_word_ratio * 100
                    : undefined,
                speaking_metrics: finalReport.speaking_metrics || {},
                filler_word_ratio: finalReport.filler_word_ratio
              },
              highlights: {
                thumbnail_url: progressData.thumbnail_url || finalReport.thumbnail_url || null
              },
              created_at: finalReport.timestamp || new Date().toISOString()
            }

            if (!user) {
              persistGuestSession(sessionStub)
            } else {
              setAnalysisHistory((prev) => {
                const filtered = prev.filter((session) => session.session_id !== sessionStub.session_id)
                return [sessionStub, ...filtered]
              })
              const fetchedSessions = await loadAnalysisHistory()
              if (!Array.isArray(fetchedSessions) || fetchedSessions.length === 0) {
                setAnalysisHistory((prev) => {
                  const filtered = prev.filter((session) => session.session_id !== sessionStub.session_id)
                  return [sessionStub, ...filtered]
                })
              }
            }

            // Navigate to video analysis page
            navigate(`/analysis/${uploadData.session_id}`)
          }
        } catch (error) {
          console.error('Error fetching progress:', error)
          clearInterval(progressInterval)
          setProcessing(false)
          alert(`Error checking progress: ${error.message}`)
        }
      }, 1000)

    } catch (error) {
      console.error('Upload error:', error)
      setProcessing(false)

      if (error.message?.includes('Failed to fetch') || error.message?.includes('ERR_CONNECTION_REFUSED')) {
        alert('Cannot connect to server. Please ensure the backend is running on http://localhost:8000')
      } else if (error.message?.includes('404')) {
        alert('Upload endpoint not found. Please check the server configuration.')
      } else {
        alert(`Upload failed: ${error.message || 'Please try again.'}`)
      }
    }
  }

  const handleProjectConfirm = () => {
    const trimmedName = (projectNameDraft || '').trim()
    if (!trimmedName) {
      setProjectNameError('Please enter a project name before starting analysis.')
      return
    }
    setProjectName(trimmedName)
    setProjectNameError('')
    setShowProjectModal(false)
    uploadFile(trimmedName)
  }

  const handleProjectCancel = () => {
    closeProjectModal()
  }

  const handleGoogleSignIn = async () => {
    try {
      await signInWithGoogle()
    } catch (error) {
      console.error('Sign in error:', error)
      alert('Could not sign in. Please try again or use guest mode.')
    }
  }

  const handleLogout = async () => {
    try {
      await logout()
      setShowLogin(true)
    } catch (error) {
      console.error('Logout error:', error)
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date'
    try {
      const date = new Date(dateString)
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      })
    } catch {
      return dateString
    }
  }

  const formatDateTimeParts = (dateString) => {
    if (!dateString) return { date: '—', time: '', full: '' }
    try {
      const date = new Date(dateString)
      if (Number.isNaN(date.getTime())) throw new Error('Invalid date')

      const dateLabel = date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      })
      const timeLabel = date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit'
      })
      const fullLabel = date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        second: '2-digit'
      })

      return {
        date: dateLabel,
        time: timeLabel,
        full: fullLabel
      }
    } catch {
      const fallback = String(dateString)
      return { date: fallback, time: '', full: fallback }
    }
  }

  const getScoreColor = (score) => {
    if (!score || Number.isNaN(Number(score))) return 'var(--text-tertiary)'
    if (score >= 80) return 'var(--accent-success)'
    if (score >= 60) return 'var(--accent-warning)'
    return 'var(--accent-danger)'
  }

  const formatDuration = (seconds, fallbackLabel) => {
    if (fallbackLabel) {
      const normalized = String(fallbackLabel).trim()
      if (normalized && !/^unknown$/i.test(normalized) && normalized !== '—') {
        return normalized
      }
    }
    const totalSeconds = Number(seconds)
    if (!Number.isFinite(totalSeconds)) return '0:00'
    if (totalSeconds < 0) return '0:00'
    const mins = Math.floor(totalSeconds / 60)
    const secs = Math.max(0, Math.round(totalSeconds % 60))
    const hours = Math.floor(mins / 60)
    const remainingMins = mins % 60
    if (hours > 0) {
      return `${hours}:${remainingMins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
    }
    return `${remainingMins}:${secs.toString().padStart(2, '0')}`
  }

  const formatPacing = (value) => {
    if (value === null || value === undefined) return '—'
    const numeric = Number(value)
    if (!Number.isFinite(numeric)) return '—'
    if (numeric > 120 && numeric < 240) {
      return `${Math.round(numeric)} wpm`
    }
    return Math.round(numeric)
  }

  const formatType = (type) => {
    if (!type) return '—'
    const normalized = type.toString().toLowerCase()
    if (normalized === 'audio') return 'Audio'
    if (normalized === 'video') return 'Video'
    return normalized.charAt(0).toUpperCase() + normalized.slice(1)
  }

  const applyRetentionFilter = useCallback((sessions) => {
    if (!Array.isArray(sessions)) return []
    if (retentionFilter === 'recent') {
      return sessions.slice(0, 2)
    }

    const now = Date.now()
    const parseToMs = (value) => {
      if (!value) return null
      const ms = Date.parse(value)
      return Number.isNaN(ms) ? null : ms
    }

    if (retentionFilter === 'all') {
      return sessions
    }

    const days = Number(retentionFilter)
    if (!Number.isNaN(days)) {
      const horizon = now - days * 24 * 60 * 60 * 1000
      return sessions.filter((session) => {
        const timestamp = parseToMs(session.created_at)
        return timestamp === null ? true : timestamp >= horizon
      })
    }

    return sessions
  }, [retentionFilter])

  const filteredHistory = useMemo(() => {
    const baseList = applyRetentionFilter(analysisHistory)
    if (!searchTerm.trim()) return baseList
    const query = searchTerm.trim().toLowerCase()
    return baseList.filter((session) => {
      const filename = session.file_name || ''
      const typeLabel = session.file_type || ''
      return (
        filename.toLowerCase().includes(query) ||
        typeLabel.toLowerCase().includes(query) ||
        (session.session_id || '').toLowerCase().includes(query)
      )
    })
  }, [analysisHistory, applyRetentionFilter, searchTerm])

  const toggleSessionSelection = (session) => {
    setSelectedSessions((prev) => {
      const exists = prev.find((item) => item.session_id === session.session_id)
      if (exists) {
        return prev.filter((item) => item.session_id !== session.session_id)
      }
      if (prev.length >= 2) {
        return [prev[1], session]
      }
      return [...prev, session]
    })
  }

  const clearSelection = () => setSelectedSessions([])

  const handleDeleteSession = async (session, event) => {
    event.stopPropagation()
    if (!session?.session_id) return

    if (!user?.email) {
      const confirmedGuest = window.confirm('Remove this analysis from your local history?')
      if (!confirmedGuest) return

      setAnalysisHistory((prev) => prev.filter((item) => item.session_id !== session.session_id))
      const updated = getStoredGuestHistory().filter((item) => item.session_id !== session.session_id)
      writeGuestHistory(updated)
      setSelectedSessions((prev) => prev.filter((item) => item.session_id !== session.session_id))
      return
    }

    const confirmed = window.confirm('Delete this analysis from your history?')
    if (!confirmed) return

    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/${session.session_id}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: user.email,
          uid: user.uid
        })
      })

      if (!response.ok) {
        let message = 'Failed to delete analysis.'
        try {
          const payload = await response.json()
          if (payload?.detail) message = payload.detail
        } catch (err) {
          // ignore JSON parse errors
        }
        alert(message)
        return
      }

      setAnalysisHistory((prev) => prev.filter((item) => item.session_id !== session.session_id))
      setSelectedSessions((prev) => prev.filter((item) => item.session_id !== session.session_id))
    } catch (error) {
      console.error('Delete session error:', error)
      alert('Could not delete this analysis. Please try again.')
    }
  }

  const handleDeleteAllSessions = async () => {
    if (!analysisHistory.length) {
      alert('No saved analyses to remove.')
      return
    }

    if (!user?.email) {
      const confirmationGuest = window.confirm('This will clear all locally saved guest analyses. Continue?')
      if (!confirmationGuest) return
      writeGuestHistory([])
      setAnalysisHistory([])
      setSelectedSessions([])
      setSearchTerm('')
      setRetentionFilter('recent')
      setLoadingHistory(false)
      return
    }

    const confirmation = window.confirm(
      'This will permanently remove every saved analysis, including generated reports and thumbnails. Continue?'
    )
    if (!confirmation) return

    setBulkDeleting(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          email: user.email,
          uid: user.uid
        })
      })

      if (!response.ok) {
        let message = 'Failed to delete history.'
        try {
          const payload = await response.json()
          if (payload?.detail) message = payload.detail
        } catch (err) {
          // ignore parsing issues
        }
        alert(message)
        return
      }

      setAnalysisHistory([])
      setSelectedSessions([])
      setSearchTerm('')
      setRetentionFilter('recent')
      setLoadingHistory(false)
    } catch (error) {
      console.error('Delete history error:', error)
      alert('Could not delete history. Please try again.')
    } finally {
      setBulkDeleting(false)
    }
  }

  const sortedSelection = useMemo(() => {
    if (selectedSessions.length !== 2) return null
    const parseDate = (value) => {
      if (!value) return 0
      const timestamp = Date.parse(value)
      return Number.isNaN(timestamp) ? 0 : timestamp
    }
    return [...selectedSessions].sort((a, b) => parseDate(a.created_at) - parseDate(b.created_at))
  }, [selectedSessions])

  const renderComparisonMetric = (label, key) => {
    if (!sortedSelection) return null
    const [baseline, latest] = sortedSelection
    const baselineValue = Number(baseline?.[key] || 0)
    const latestValue = Number(latest?.[key] || 0)
    const maxValue = Math.max(baselineValue, latestValue, 1)
    const delta = latestValue - baselineValue
    const deltaLabel = `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}`
    const deltaClass = delta > 0 ? 'delta-positive' : delta < 0 ? 'delta-negative' : 'delta-neutral'

    return (
      <div className="comparison-metric" key={key}>
        <div className="comparison-metric-header">
          <span>{label}</span>
          <span className={`comparison-delta ${deltaClass}`}>{deltaLabel}</span>
        </div>
        <div className="comparison-bars">
          <div className="comparison-bar">
            <div
              className="comparison-bar-fill baseline"
              style={{ width: `${(baselineValue / maxValue) * 100}%` }}
            />
          </div>
          <div className="comparison-bar">
            <div
              className="comparison-bar-fill latest"
              style={{ width: `${(latestValue / maxValue) * 100}%` }}
            />
          </div>
        </div>
        <div className="comparison-values">
          <span>Baseline: {baselineValue.toFixed(1)}</span>
          <span>Latest: {latestValue.toFixed(1)}</span>
        </div>
      </div>
    )
  }

  const retentionOptions = [
    { value: 'recent', label: 'Last 2 sessions' },
    { value: '30', label: 'Last 30 days' },
    { value: '90', label: 'Last 90 days' },
    { value: '365', label: 'Last 12 months' },
    { value: 'all', label: 'All history' }
  ]

  if (showLogin) {
    return (
      <motion.div
        className="auth-screen"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <div className="auth-bg-gradient" />
        <motion.div
          className="auth-card"
          initial={{ scale: 0.9, y: 20 }}
          animate={{ scale: 1, y: 0 }}
          transition={{ type: "spring", duration: 0.5 }}
        >
          <h1 className="auth-title">Face2Phrase</h1>
          <p className="auth-subtitle">AI Communication Analysis</p>

          <motion.button
            className="auth-btn google-btn"
            onClick={handleGoogleSignIn}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <i className="fab fa-google"></i>
            Sign in with Google
          </motion.button>

          <motion.button
            className="auth-btn guest-btn"
            onClick={() => setShowLogin(false)}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <i className="fas fa-user"></i>
            Continue as Guest
          </motion.button>
        </motion.div>
      </motion.div>
    )
  }

  return (
    <>
      <div className="dashboard-container">
        {/* Header */}
        <header className="dashboard-header">
          <div>
            <h1 className="header-title">Face2Phase</h1>
            <p className="header-subtitle">AI Communication Analysis</p>
          </div>
          <div className="header-actions">
            <div className="user-profile">
              {user?.displayName?.[0] || user?.email?.[0] || 'G'}
            </div>
            <button className="logout-btn" onClick={handleLogout}>
              <i className="fas fa-sign-out-alt" /> Logout
            </button>
          </div>
        </header>

        <div className="dashboard-content">
          {/* Upload Section */}
          <motion.div
            className="upload-section"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,video/*"
              onChange={handleFileChange}
              className="file-input"
            />
            <motion.div
              className={`upload-area ${file ? 'has-file' : ''}`}
              onClick={(e) => {
                if (file) return
                fileInputRef.current?.click()
              }}
              whileHover={{ scale: file ? 1 : 1.02 }}
              whileTap={{ scale: file ? 1 : 0.98 }}
            >
              <div className="upload-label">
                <i className="fas fa-upload upload-icon" />
                {file ? 'File Selected' : 'Upload File'}
              </div>
              <div className="upload-text">
                {file ? file.name : 'Click to select audio or video file'}
              </div>
              <div
                className="recording-controls"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="recording-header">
                  <i className="fas fa-dot-circle" />
                  <span>Or record directly</span>
                </div>
                <div className="recording-buttons">
                  <button
                    type="button"
                    className="record-btn"
                    disabled={!supportsRecording || isRecording || processing}
                    onClick={(e) => {
                      e.preventDefault()
                      startRecording('video')
                    }}
                  >
                    <i className="fas fa-video" />
                    {isRecording && recordingType === 'video' ? 'Recording…' : 'Record Video'}
                  </button>
                  <button
                    type="button"
                    className="record-btn"
                    disabled={!supportsRecording || isRecording || processing}
                    onClick={(e) => {
                      e.preventDefault()
                      startRecording('audio')
                    }}
                  >
                    <i className="fas fa-microphone" />
                    {isRecording && recordingType === 'audio' ? 'Recording…' : 'Record Audio'}
                  </button>
                  {isRecording && (
                    <>
                      <button
                        type="button"
                        className="record-btn stop"
                        onClick={(e) => {
                          e.preventDefault()
                          stopRecording()
                        }}
                      >
                        <i className="fas fa-stop" />
                        Stop & Save
                      </button>
                      <button
                        type="button"
                        className="record-btn ghost"
                        onClick={(e) => {
                          e.preventDefault()
                          cancelRecording()
                        }}
                      >
                        <i className="fas fa-times" />
                        Discard
                      </button>
                    </>
                  )}
                </div>
                {recordingType === 'video' && previewStream && (
                  <div className="recording-preview-wrapper">
                    <video
                      ref={previewVideoRef}
                      className="recording-preview"
                      autoPlay
                      muted
                      playsInline
                    />
                  </div>
                )}
                {recordingType === 'audio' && isRecording && (
                  <div className="recording-audio-indicator">
                    <i className="fas fa-wave-square" />
                    Listening…
                  </div>
                )}
                {isRecording && (
                  <div className="recording-status">
                    <span className="recording-indicator" />
                    Recording {recordingType} · {formatRecordingTime(recordingDuration)}
                  </div>
                )}
                {!supportsRecording && (
                  <div className="recording-error">
                    Browser recording is not supported here. Please upload a file instead.
                  </div>
                )}
                {recordingError && (
                  <div className="recording-error">
                    {recordingError}
                  </div>
                )}
              </div>
              {file && (
                <motion.button
                  className="analyze-button"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    if (processing) return
                    openProjectModal()
                  }}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  whileHover={{ scale: processing ? 1 : 1.05 }}
                  whileTap={{ scale: processing ? 1 : 0.95 }}
                  onMouseDown={(e) => e.stopPropagation()}
                  disabled={processing}
                >
                  Start Analysis
                </motion.button>
              )}
            </motion.div>
          </motion.div>

          {/* Processing State */}
          {processing && <LoadingScreen progress={progress} />}

          <div className="history-section">
            <h2 className="section-title">Previous Analyses</h2>

            {/* FIX 2: Hide history for guest users, show sign-in prompt */}
            {isGuest ? (
              <div className="guest-history-message" style={{ textAlign: 'center', padding: '80px 24px' }}>
                <i className="fas fa-lock" style={{ fontSize: '48px', marginBottom: '16px', opacity: 0.5 }}></i>
                <h3 style={{ marginBottom: '8px' }}>Sign in to view your analysis history</h3>
                <p style={{ opacity: 0.7, marginBottom: '24px' }}>
                  Create an account to save and sync your analysis sessions across all your devices.
                </p>
                <button
                  className="analyze-button"
                  onClick={handleGoogleSignIn}
                  style={{ padding: '12px 32px' }}
                >
                  <i className="fab fa-google" style={{ marginRight: '8px' }}></i>
                  Sign in with Google
                </button>
              </div>
            ) : (
              <>

                <div className="history-controls">
                  <div className="history-search">
                    <i className="fas fa-search" />
                    <input
                      type="text"
                      placeholder="Search by title, type, or session ID"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                    />
                  </div>
                  <div className="history-retention">
                    <label htmlFor="retention-filter">Retention</label>
                    <select
                      id="retention-filter"
                      value={retentionFilter}
                      onChange={(e) => setRetentionFilter(e.target.value)}
                    >
                      {retentionOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="history-insights">
                    <span>{filteredHistory.length} sessions</span>
                    {selectedSessions.length > 0 && (
                      <button className="clear-selection-btn" onClick={clearSelection}>
                        <i className="fas fa-times" /> Clear selection
                      </button>
                    )}
                    {analysisHistory.length > 0 && (
                      <button
                        className="clear-history-btn"
                        onClick={handleDeleteAllSessions}
                        disabled={bulkDeleting}
                      >
                        <i className="fas fa-trash-alt" /> {bulkDeleting ? 'Deleting…' : 'Delete all'}
                      </button>
                    )}
                  </div>
                </div>

                {sortedSelection && (
                  <div className="comparison-panel">
                    <div className="comparison-header">
                      <div>
                        <h3>Session Comparison</h3>
                        <p>
                          Comparing <strong>{formatDate(sortedSelection[0].created_at)}</strong> and{' '}
                          <strong>{formatDate(sortedSelection[1].created_at)}</strong>
                        </p>
                      </div>
                      <div className="comparison-legend">
                        <span className="legend baseline">
                          <span /> Baseline
                        </span>
                        <span className="legend latest">
                          <span /> Latest
                        </span>
                      </div>
                    </div>
                    <div className="comparison-grid">
                      {renderComparisonMetric('Overall Score', 'overall_score')}
                      {renderComparisonMetric('Voice Confidence', 'voice_confidence')}
                      {renderComparisonMetric('Facial Confidence', 'facial_confidence')}
                      {renderComparisonMetric('Vocabulary Score', 'vocabulary_score')}
                    </div>
                  </div>
                )}

                {loadingHistory ? (
                  <div className="loading-history">
                    <i className="fas fa-spinner fa-spin" /> Loading...
                  </div>
                ) : filteredHistory.length === 0 ? (
                  <div className="empty-history">
                    <i className="fas fa-folder-open" />
                    <p>No previous analyses found</p>
                    <p className="empty-hint">Upload a file to get started</p>
                  </div>
                ) : (
                  <>
                    <div className="history-table" role="table" aria-label="Previous analyses">
                      <div className="history-table-header" role="row">
                        <div className="history-col title" role="columnheader">
                          <span>Title</span>
                          <i className="fas fa-sort-up"></i>
                        </div>
                        <div className="history-col created" role="columnheader">
                          <span>Created</span>
                          <i className="fas fa-sort"></i>
                        </div>
                        <div className="history-col score" role="columnheader">
                          <span>Overall</span>
                        </div>
                        <div className="history-col type" role="columnheader">
                          <span>Type</span>
                        </div>
                        <div className="history-col duration" role="columnheader">
                          <span>Duration</span>
                        </div>
                        <div className="history-col voice" role="columnheader">
                          <span>Voice</span>
                        </div>
                        <div className="history-col facial" role="columnheader">
                          <span>Facial</span>
                        </div>
                        <div className="history-col actions" role="columnheader" aria-label="Actions" />
                      </div>

                      <div className="history-table-body">
                        {filteredHistory.map((analysis, idx) => {
                          const isSelected = selectedSessions.some((session) => session.session_id === analysis.session_id)
                          const scoreDisplay =
                            analysis.overall_score || analysis.overall_score === 0
                              ? Math.round(Number(analysis.overall_score))
                              : null
                          const typeIcon = (analysis.file_type || '').toLowerCase() === 'video' ? 'video' : 'microphone'
                          const createdInfo = formatDateTimeParts(analysis.created_at)
                          const displayTitle =
                            analysis.display_title ||
                            analysis.project_name ||
                            analysis.file_name ||
                            analysis.title ||
                            'Untitled Session'

                          return (
                            <motion.div
                              key={analysis.session_id || analysis.id}
                              className="history-row"
                              role="row"
                              data-selected={isSelected}
                              initial={{ opacity: 0, y: 12 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: Math.min(idx * 0.04, 0.4) }}
                              onClick={() => navigate(`/analysis/${analysis.session_id || analysis.id}`)}
                            >
                              <div className="history-cell title" role="cell">
                                <div className="history-thumb">
                                  {analysis.thumbnail_url ? (
                                    <img src={analysis.thumbnail_url} alt="" />
                                  ) : (
                                    <span className="thumb-icon">
                                      <i className={`fas fa-${typeIcon}`} />
                                    </span>
                                  )}
                                </div>
                                <div className="history-title-block">
                                  <span className="history-title-text">{displayTitle}</span>
                                  <span className="history-subtext">
                                    {(analysis.session_id || analysis.id || '').slice(0, 8) || '—'}
                                  </span>
                                </div>
                              </div>

                              <div className="history-cell created" role="cell">
                                <div className="date-stack" title={createdInfo.full || undefined}>
                                  <span className="date-primary">{createdInfo.date}</span>
                                  {createdInfo.time && <span className="date-secondary">{createdInfo.time}</span>}
                                </div>
                              </div>

                              <div className="history-cell score" role="cell">
                                <span style={{ color: getScoreColor(scoreDisplay) }}>
                                  {scoreDisplay !== null ? scoreDisplay : '—'}
                                </span>
                              </div>

                              <div className="history-cell type" role="cell">
                                <span className="type-pill">{formatType(analysis.file_type)}</span>
                              </div>

                              <div className="history-cell duration" role="cell">
                                <span>
                                  {formatDuration(
                                    analysis.duration ||
                                    analysis.duration_seconds ||
                                    analysis.speaking_metrics?.total_duration ||
                                    analysis.total_duration ||
                                    0
                                  )}
                                </span>
                              </div>

                              <div className="history-cell voice" role="cell">
                                <span style={{ color: getScoreColor(analysis.voice_confidence) }}>
                                  {analysis.voice_confidence != null && !isNaN(analysis.voice_confidence)
                                    ? Math.round(Number(analysis.voice_confidence))
                                    : '—'}
                                </span>
                              </div>

                              <div className="history-cell facial" role="cell">
                                <span style={{ color: getScoreColor(analysis.facial_confidence) }}>
                                  {analysis.facial_confidence != null && !isNaN(analysis.facial_confidence)
                                    ? Math.round(Number(analysis.facial_confidence))
                                    : '—'}
                                </span>
                              </div>

                              <div className="history-cell actions" role="cell">
                                <button
                                  className={`row-select-btn ${isSelected ? 'selected' : ''}`}
                                  title={isSelected ? 'Remove from comparison' : 'Compare session'}
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    toggleSessionSelection(analysis)
                                  }}
                                >
                                  <i className={isSelected ? 'fas fa-check' : 'fas fa-plus'} />
                                </button>
                                <button
                                  className="row-delete-btn"
                                  title="Delete analysis"
                                  onClick={(e) => handleDeleteSession(analysis, e)}
                                >
                                  <i className="fas fa-trash" />
                                </button>
                                <button
                                  className="row-menu-btn"
                                  title="More options"
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    navigate(`/analysis/${analysis.session_id || analysis.id}`)
                                  }}
                                >
                                  <i className="fas fa-ellipsis-vertical" />
                                </button>
                              </div>
                            </motion.div>
                          )
                        })}
                      </div>
                    </div>

                    {filteredHistory.length > 0 && (
                      <div className="history-table-footer">
                        Displaying items 1 - {filteredHistory.length} of {filteredHistory.length}
                      </div>
                    )}
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {showProjectModal && (
        <div className="project-modal-backdrop" onClick={handleProjectCancel}>
          <div
            className="project-modal"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="project-modal-title"
          >
            <h3 id="project-modal-title">Name Your Project</h3>
            <p className="project-modal-description">
              Give this analysis a descriptive name. You’ll see it in your history and reports.
            </p>
            <input
              type="text"
              className={`project-modal-input ${projectNameError ? 'has-error' : ''}`}
              value={projectNameDraft}
              onChange={(e) => {
                setProjectNameDraft(e.target.value)
                if (projectNameError) setProjectNameError('')
              }}
              placeholder="e.g. Sales Pitch Dry Run"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  handleProjectConfirm()
                } else if (e.key === 'Escape') {
                  e.preventDefault()
                  handleProjectCancel()
                }
              }}
            />
            {projectNameError && <p className="project-modal-error">{projectNameError}</p>}
            <div className="project-modal-actions">
              <button type="button" className="modal-btn secondary" onClick={handleProjectCancel}>
                Cancel
              </button>
              <button
                type="button"
                className="modal-btn primary"
                onClick={handleProjectConfirm}
                disabled={processing}
              >
                Start Analysis
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default Dashboard
