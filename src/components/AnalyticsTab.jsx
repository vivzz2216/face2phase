import { useState, useCallback, useMemo } from 'react'
import { motion } from 'framer-motion'
import './AnalyticsTab.css'

const AnalyticsTab = ({ reportData }) => {
  const [activeSubTab, setActiveSubTab] = useState('word-choice')
  const [showWeakWordsModal, setShowWeakWordsModal] = useState(false)

  if (!reportData) {
    return <div className="analytics-empty">No analytics data available</div>
  }

  // Check if this is audio-only (no video/facial analysis)
  const isAudioOnly = reportData.is_audio_only || reportData.file_type === 'audio' || false
  console.log('📊 AnalyticsTab - isAudioOnly:', isAudioOnly, 'file_type:', reportData.file_type)

  // Get word analysis data from report (preferred) or calculate from report data
  const wordAnalysis = reportData.word_analysis || {}
  const weakWordsData = wordAnalysis.weak_words || {}
  const fillerWordsData = wordAnalysis.filler_words || {}
  const vocabularyData = wordAnalysis.vocabulary || {}

  // Get ACTUAL filler analysis from audio_results (most reliable)
  const fillerAnalysis = reportData.filler_analysis || {}
  const actualFillerCount = fillerAnalysis.total_fillers || fillerWordsData.filler_count || reportData.filler_word_count || 0
  const actualFillerRatio = fillerAnalysis.filler_ratio || 0
  // Filter out [acoustic] entries - show only fillers from transcript
  const rawFillerBreakdown = fillerAnalysis.filler_breakdown || fillerWordsData.filler_breakdown || {}
  const actualFillerBreakdown = Object.fromEntries(
    Object.entries(rawFillerBreakdown).filter(([word]) => !word.startsWith('[acoustic]'))
  )
  const acousticEventsRaw = Array.isArray(fillerAnalysis.acoustic_events) ? fillerAnalysis.acoustic_events : []
  const textModelEventsRaw = Array.isArray(fillerAnalysis.text_model_fillers) ? fillerAnalysis.text_model_fillers : []
  const normalizedTextEvents = textModelEventsRaw
    .filter((event) => typeof event?.start === 'number')
    .map((event) => {
      const start = event.start
      const end = typeof event.end === 'number' ? event.end : start + 0.3
      const duration = Math.max(0, (typeof event.duration === 'number' ? event.duration : end - start))
      return {
        label: event.token_original || event.label || 'filler',
        start,
        end,
        duration,
        confidence: typeof event.score === 'number' ? event.score : (event.confidence || 0),
        method: event.method || 'text_model'
      }
    })
  const detectedFillerEvents = [
    ...acousticEventsRaw,
    ...normalizedTextEvents
  ]
  const acousticFillerCountRaw = fillerAnalysis.acoustic_filler_count || fillerAnalysis.acoustic_fillers || acousticEventsRaw.length || 0
  const textModelFillerCount = fillerAnalysis.text_model_filler_count || textModelEventsRaw.length || 0
  const acousticFillerCount = detectedFillerEvents.length || (acousticFillerCountRaw + textModelFillerCount)

  // Get actual total words from multiple sources
  const totalWords = vocabularyData.total_words ||
    reportData.total_words ||
    reportData.speaking_metrics?.total_words ||
    (reportData.transcript ? reportData.transcript.split().length : 0) ||
    1

  // Calculate filler percentage from ACTUAL data
  const fillerPercentage = actualFillerRatio > 0
    ? (actualFillerRatio * 100).toFixed(1)
    : (totalWords > 0 ? ((actualFillerCount / totalWords) * 100).toFixed(1) : '0.0')

  // Weak words data - use ACTUAL data
  const weakWordsPercentage = weakWordsData.weak_word_percentage ||
    (weakWordsData.weak_word_count && totalWords > 0
      ? ((weakWordsData.weak_word_count / totalWords) * 100).toFixed(1)
      : '0.0')
  const weakWordBreakdown = weakWordsData.weak_word_breakdown || {}
  const weakWordsFound = weakWordsData.weak_words_found || []

  // Mumbling detection from actual analysis
  const mumblingCount = fillerAnalysis.mumbling_count || acousticFillerCount || 0
  const mumblingDetected = mumblingCount > 0 || fillerAnalysis.mumbling_instances?.length > 0 || acousticFillerCount > 0

  // Vocabulary metrics
  const vocabularyRichness = vocabularyData.vocabulary_richness || reportData.vocabulary_richness || 0
  const uniqueWords = vocabularyData.unique_words || reportData.unique_words || 0
  const vocabularySuggestions = vocabularyData.suggestions || []

  // Calculate conciseness excess based on speaking rate and word count
  const speakingRate = reportData.speaking_rate_wpm || 150
  const idealRate = 150
  const rateExcess = speakingRate > idealRate ? ((speakingRate - idealRate) / idealRate * 100) : 0
  const wordCountExcess = totalWords > 200 ? ((totalWords - 200) / 200 * 100) : 0
  const excessPercentage = Math.round((rateExcess + wordCountExcess) / 2)
  const eyeContactRaw = reportData.avg_eye_contact
  const eyeContactPercent = typeof eyeContactRaw === 'number'
    ? (eyeContactRaw <= 1 ? eyeContactRaw * 100 : eyeContactRaw)
    : null
  const pauseAnalysis = reportData.pause_analysis || {}
  const pauseSummary = Object.keys(reportData.pause_summary || {}).length
    ? reportData.pause_summary
    : pauseAnalysis
  const pausesDetailed = Array.isArray(reportData.pauses_detailed) ? reportData.pauses_detailed : []
  const pauseCount = typeof pauseSummary?.total_pauses === 'number'
    ? pauseSummary.total_pauses
    : (pausesDetailed.length || reportData.pause_count || 0)

  const audioAnalytics = reportData.audio_analytics || reportData.advanced_audio_metrics || {}
  const visualAnalyticsRaw = reportData.visual_analytics || {}
  const baseTensionSummary = visualAnalyticsRaw.tension_summary || reportData.tension_summary || {}
  const derivedTensionSummary = {
    ...baseTensionSummary,
    tension_percentage: typeof baseTensionSummary.tension_percentage === 'number'
      ? baseTensionSummary.tension_percentage
      : (typeof reportData.tension_percentage === 'number' ? reportData.tension_percentage : null),
    avg_eye_contact_pct: typeof baseTensionSummary.avg_eye_contact_pct === 'number'
      ? baseTensionSummary.avg_eye_contact_pct
      : (typeof eyeContactPercent === 'number' ? eyeContactPercent : null),
    eye_contact_stability: baseTensionSummary.eye_contact_stability ?? visualAnalyticsRaw.eye_contact_stability ?? null
  }
  const visualAnalytics = {
    tension_summary: derivedTensionSummary,
    emotion_timeline_smoothed: visualAnalyticsRaw.emotion_timeline_smoothed || reportData.emotion_timeline_smoothed || [],
  }
  const textAnalytics = reportData.text_analytics || reportData.advanced_text_metrics || {}

  return (
    <>
      <div className="analytics-tab">
        <div className="analytics-subtabs">
          <button
            className={`subtab-btn ${activeSubTab === 'word-choice' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('word-choice')}
          >
            Word Choice
          </button>
          <button
            className={`subtab-btn ${activeSubTab === 'delivery' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('delivery')}
          >
            Delivery
          </button>
          <button
            className={`subtab-btn ${activeSubTab === 'advanced' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('advanced')}
          >
            Advanced
          </button>
        </div>

        <div className="analytics-content">
          {activeSubTab === 'word-choice' && (
            <WordChoiceAnalytics
              fillerPercentage={fillerPercentage}
              weakWordsPercentage={weakWordsPercentage}
              excessPercentage={excessPercentage}
              reportData={reportData}
              totalWords={totalWords}
              weakWordBreakdown={weakWordBreakdown}
              weakWordsFound={weakWordsFound}
              fillerBreakdown={actualFillerBreakdown}
              mumblingDetected={mumblingDetected}
              mumblingCount={mumblingCount}
              acousticEvents={detectedFillerEvents}
              acousticCount={acousticFillerCount}
              vocabularyRichness={vocabularyRichness}
              uniqueWords={uniqueWords}
              vocabularySuggestions={vocabularySuggestions}
            />
          )}
          {activeSubTab === 'delivery' && (
            <DeliveryAnalytics
              speakingRate={speakingRate}
              pauseCount={pauseCount}
              eyeContactPercent={eyeContactPercent}
              pauseSummary={pauseSummary}
              pausesDetailed={pausesDetailed}
              isAudioOnly={isAudioOnly}
            />
          )}
          {activeSubTab === 'advanced' && (
            <AdvancedAnalytics
              audioAnalytics={audioAnalytics}
              visualAnalytics={visualAnalytics}
              textAnalytics={textAnalytics}
              structureMetrics={reportData?.text_analysis?.structure_metrics || {}}
              contentMetrics={reportData?.text_analysis?.content_metrics || {}}
              voiceConfidence={reportData?.voice_confidence_score ?? reportData?.voice_confidence ?? null}
              transcript={reportData?.transcript || ''}
              isAudioOnly={isAudioOnly}
            />
          )}
        </div>
      </div>

      {/* Weak Words Modal */}
      {showWeakWordsModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10000,
            padding: '20px'
          }}
          onClick={() => setShowWeakWordsModal(false)}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-card)',
              borderRadius: '12px',
              padding: '24px',
              maxWidth: '600px',
              maxHeight: '80vh',
              overflowY: 'auto',
              boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
              border: '1px solid var(--border-color)'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <h2 style={{ margin: 0, fontSize: '1.5rem', color: 'var(--text-primary)', fontWeight: 700 }}>What are Weak Words?</h2>
              <button
                onClick={() => setShowWeakWordsModal(false)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  fontSize: '1.5rem',
                  cursor: 'pointer',
                  color: 'var(--text-secondary)',
                  padding: '0',
                  width: '32px',
                  height: '32px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
              >
                ×
              </button>
            </div>

            <div style={{ color: 'var(--text-primary)', lineHeight: 1.6 }}>
              <p style={{ marginBottom: '16px', fontSize: '1rem' }}>
                <strong>Weak words</strong> are words that make your speech less impactful, less precise, or less professional. They include:
              </p>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--text-primary)', fontWeight: 600 }}>
                  Uncertainty Words
                </h3>
                <ul style={{ paddingLeft: '20px', marginBottom: '12px' }}>
                  <li><strong>"just"</strong> - Use more specific time references or remove entirely</li>
                  <li><strong>"really"</strong> - Use stronger, more precise adjectives</li>
                  <li><strong>"very"</strong> - Replace with more descriptive words (e.g., "very good" → "excellent")</li>
                  <li><strong>"maybe", "perhaps", "probably"</strong> - Be more direct or state confidence level</li>
                  <li><strong>"kind of", "sort of"</strong> - Use "somewhat" or "to some extent" for clarity</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--text-primary)', fontWeight: 600 }}>
                  Vague Qualifiers
                </h3>
                <ul style={{ paddingLeft: '20px', marginBottom: '12px' }}>
                  <li><strong>"thing", "stuff", "things"</strong> - Use specific nouns (e.g., "thing" → "concept", "element")</li>
                  <li><strong>"something", "anything", "everything"</strong> - Be more specific about what you're referring to</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: '12px', color: 'var(--text-primary)', fontWeight: 600 }}>
                  Weak Verbs
                </h3>
                <ul style={{ paddingLeft: '20px', marginBottom: '12px' }}>
                  <li><strong>"got", "get"</strong> - Use "have", "received", "obtained" instead</li>
                  <li><strong>"gonna", "wanna"</strong> - Use "going to" or "want to"</li>
                </ul>
              </div>

              <div style={{ marginBottom: '20px', padding: '12px', backgroundColor: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                  <strong>💡 Tip:</strong> Replacing weak words with stronger, more specific language makes your presentations more impactful and professional. Aim to keep weak words below 5% of your total words.
                </p>
              </div>

              <button
                onClick={() => setShowWeakWordsModal(false)}
                style={{
                  marginTop: '20px',
                  padding: '10px 24px',
                  backgroundColor: 'var(--accent-primary)',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '0.95rem',
                  fontWeight: 600,
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => e.target.style.opacity = '0.9'}
                onMouseLeave={(e) => e.target.style.opacity = '1'}
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}

const WordChoiceAnalytics = ({
  fillerPercentage,
  weakWordsPercentage,
  excessPercentage,
  reportData,
  totalWords,
  weakWordBreakdown = {},
  weakWordsFound = [],
  fillerBreakdown = {},
  mumblingDetected = false,
  acousticEvents = [],
  acousticCount = 0,
  mumblingCount = 0,
  vocabularyRichness = 0,
  uniqueWords = 0,
  vocabularySuggestions = []
}) => {
  const [expandedFiller, setExpandedFiller] = useState(false)
  const [expandedWeak, setExpandedWeak] = useState(false)
  const [expandedConciseness, setExpandedConciseness] = useState(false)
  const [expandedSentenceOpeners, setExpandedSentenceOpeners] = useState(false)
  const [expandedVocabulary, setExpandedVocabulary] = useState(false)
  const [loadingState, setLoadingState] = useState({
    filler: false,
    weak: false,
    conciseness: false,
    sentence_openers: false,
    vocabulary: false
  })
  const vocabularyData = reportData?.word_analysis?.vocabulary || {}
  const complexWordRatio = typeof vocabularyData.complex_word_ratio === 'number'
    ? vocabularyData.complex_word_ratio
    : (reportData?.complex_word_ratio || 0)
  const complexWordPct = (complexWordRatio * 100).toFixed(1)
  const complexWords = vocabularyData.complex_words || reportData?.complex_words || 0
  const advancedTerms = Array.isArray(vocabularyData.advanced_terms) ? vocabularyData.advanced_terms : []
  const vocabularyRichnessPct = vocabularyRichness ? (vocabularyRichness * 100).toFixed(1) : '0.0'
  const vocabularySuggestionsList = Array.isArray(vocabularySuggestions) ? vocabularySuggestions : []
  const totalWordCount = totalWords || vocabularyData.total_words || reportData?.total_words || 0
  const vocabularyStrength = vocabularyRichness >= 0.65 && complexWordRatio >= 0.2 && vocabularySuggestionsList.length === 0
  const vocabularyEnhancements = Array.isArray(reportData?.vocabulary_enhancements?.enhancements)
    ? reportData.vocabulary_enhancements.enhancements
    : []
  const vocabularyMessage = reportData?.vocabulary_enhancements?.message
  const vocabularyHasData = totalWordCount > 0 && uniqueWords > 0
  const transcript = reportData?.transcript || ''

  const autoVocabularyEnhancements = useMemo(() => {
    if (!transcript || transcript.length < 12) return []
    const enhancements = []
    const seen = new Set()
    const patterns = [
      {
        regex: /\b(i\s+am\s+very\s+good\s+at\s+)([a-z0-9\s\-]{2,60})/gi,
        build: (match, intro, subject) => ({
          word: `${intro}${subject}`.trim(),
          context: match.trim(),
          suggestions: [`I excel at ${subject.trim()}`],
          reason: 'Swap casual phrasing for a confident, professional verb.'
        })
      },
      {
        regex: /\b(i\s+am\s+good\s+at\s+)([a-z0-9\s\-]{2,60})/gi,
        build: (match, intro, subject) => ({
          word: `${intro}${subject}`.trim(),
          context: match.trim(),
          suggestions: [`I am proficient in ${subject.trim()}`],
          reason: 'Use precise language to signal expertise.'
        })
      },
      {
        regex: /\b(i\s+really\s+like\s+)([a-z0-9\s\-]{2,60})/gi,
        build: (match, intro, subject) => ({
          word: `${intro}${subject}`.trim(),
          context: match.trim(),
          suggestions: [`I am passionate about ${subject.trim()}`],
          reason: 'Replace filler adverbs with purposeful language.'
        })
      }
    ]

    patterns.forEach(({ regex, build }) => {
      let result
      while ((result = regex.exec(transcript)) !== null) {
        const [fullMatch, intro, subject] = result
        if (!subject) continue
        const suggestion = build(fullMatch, intro, subject)
        const key = `${suggestion.word.toLowerCase()}::${suggestion.suggestions[0]?.toLowerCase()}`
        if (seen.has(key)) {
          continue
        }
        seen.add(key)
        enhancements.push(suggestion)
      }
    })

    return enhancements.slice(0, 5)
  }, [transcript])

  const displayVocabularyEnhancements = vocabularyEnhancements.length > 0
    ? vocabularyEnhancements
    : autoVocabularyEnhancements


  const formatSeconds = (seconds) => {
    if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '--:--'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const triggerLoading = useCallback((sectionKey, shouldOpen) => {
    if (shouldOpen) {
      setLoadingState((prev) => ({ ...prev, [sectionKey]: true }))
      setTimeout(() => {
        setLoadingState((prev) => ({ ...prev, [sectionKey]: false }))
      }, 320)
    }
  }, [])

  const handleToggle = (sectionKey, setter, currentValue) => {
    const nextValue = !currentValue
    setter(nextValue)
    triggerLoading(sectionKey, nextValue)
  }

  const parseNumber = (value) => {
    if (typeof value === 'number') {
      return Number.isFinite(value) ? value : null
    }
    if (typeof value === 'string') {
      const parsed = parseFloat(value)
      return Number.isFinite(parsed) ? parsed : null
    }
    return null
  }

  const formatPercentDisplay = (value) => {
    if (value === null || Number.isNaN(value)) {
      return '—'
    }
    const digits = Math.abs(value) >= 10 ? 0 : 1
    const fixed = value.toFixed(digits)
    return `${fixed.replace(/\.0$/, '')}%`
  }

  const formatSentenceStarter = (label) => {
    if (!label) return 'Varied'
    const normalized = label.replace(/_/g, ' ').trim()
    if (!normalized) return 'Varied'
    const capitalised = normalized.replace(/\b\w/g, (char) => char.toUpperCase())
    return `"${capitalised},"`
  }

  const renderSummaryCard = (card, variant) => (
    <div key={card.key} className={`summary-card ${variant}`}>
      <span className="summary-card-title">{card.title}</span>
      <span className="summary-card-value">{card.primary}</span>
      {card.secondary && (
        <span className="summary-card-sub">{card.secondary}</span>
      )}
    </div>
  )

  const textAnalytics = reportData?.text_analytics || reportData?.advanced_text_metrics || {}
  const structureMetrics = reportData?.text_analysis?.structure_metrics || {}
  const sentenceOpeners = structureMetrics.sentence_openers || {}
  const openerPercentages = sentenceOpeners.opener_percentages || {}
  const totalSentences = sentenceOpeners.total_sentences || structureMetrics.total_sentences || 0

  const topOpenerEntry = Object.entries(openerPercentages)
    .filter(([, pct]) => typeof pct === 'number' && !Number.isNaN(pct))
    .sort((a, b) => b[1] - a[1])[0]
  const topOpenerLabel = topOpenerEntry ? topOpenerEntry[0] : null
  const topOpenerPctRaw = topOpenerEntry ? Number(topOpenerEntry[1]) : null
  const sentenceStarterPrimary = topOpenerLabel ? formatSentenceStarter(topOpenerLabel) : 'Varied'
  const sentenceStarterSecondary = topOpenerPctRaw !== null ? formatPercentDisplay(topOpenerPctRaw) : null
  const openerExamples = sentenceOpeners.opener_examples || {}
  const openerRecommendations = Array.isArray(sentenceOpeners.recommendations) ? sentenceOpeners.recommendations : []
  const sortedSentenceOpeners = Object.entries(openerPercentages)
    .filter(([, pct]) => typeof pct === 'number' && pct > 0)
    .sort((a, b) => b[1] - a[1])
  const hasSentenceOpenerData = sortedSentenceOpeners.length > 0

  const repetitionAlerts = Array.isArray(textAnalytics.repetition_alerts)
    ? textAnalytics.repetition_alerts
    : []
  const topRepetition = [...repetitionAlerts]
    .sort((a, b) => (b?.count || 0) - (a?.count || 0))[0]
  const repetitionCount = topRepetition?.count || 0
  let repetitionPctValue = null
  if (topRepetition?.percentage !== undefined && topRepetition?.percentage !== null) {
    repetitionPctValue = parseNumber(topRepetition.percentage)
  } else if (totalSentences > 0 && repetitionCount > 0) {
    repetitionPctValue = Number(((repetitionCount / totalSentences) * 100).toFixed(1))
  } else if (totalSentences > 0) {
    repetitionPctValue = 0
  }
  const repetitionSecondary = repetitionPctValue !== null ? formatPercentDisplay(repetitionPctValue) : null

  const fillerWordCount = reportData?.filler_analysis?.total_fillers
    ?? reportData?.word_analysis?.filler_words?.filler_count
    ?? reportData?.filler_word_count
    ?? 0
  const weakWordCount = reportData?.word_analysis?.weak_words?.weak_word_count
    ?? (Array.isArray(weakWordsFound) ? weakWordsFound.length : 0)
  const fillerPctNum = parseNumber(fillerPercentage)
  const weakPctNum = parseNumber(weakWordsPercentage)
  const concisenessNum = parseNumber(excessPercentage)

  const positiveHighlights = []
  const improvementHighlights = []

  const addCard = (collection, card) => {
    if (!card || !card.primary) return
    collection.push(card)
  }

  if (totalSentences > 0 && (repetitionCount > 0 || repetitionPctValue !== null)) {
    const repetitionCard = {
      key: 'repetition',
      title: 'Repetition',
      primary: `${repetitionCount} repetition${repetitionCount === 1 ? '' : 's'}`,
      secondary: repetitionSecondary
    }
    if (repetitionCount === 0 || (repetitionPctValue !== null && repetitionPctValue <= 8)) {
      addCard(positiveHighlights, repetitionCard)
    } else {
      addCard(improvementHighlights, repetitionCard)
    }
  }

  if (fillerPctNum !== null) {
    const fillerCard = {
      key: 'filler',
      title: 'Filler Words',
      primary: `${fillerWordCount} filler${fillerWordCount === 1 ? '' : 's'}`,
      secondary: formatPercentDisplay(fillerPctNum)
    }
    if (fillerPctNum > 4) {
      addCard(improvementHighlights, fillerCard)
    } else {
      addCard(positiveHighlights, fillerCard)
    }
  }

  if (weakPctNum !== null) {
    const weakCard = {
      key: 'weak',
      title: 'Weak Words',
      primary: `${weakWordCount} weak word${weakWordCount === 1 ? '' : 's'}`,
      secondary: formatPercentDisplay(weakPctNum)
    }
    if (weakPctNum > 3) {
      addCard(improvementHighlights, weakCard)
    } else {
      addCard(positiveHighlights, weakCard)
    }
  }

  if (concisenessNum !== null) {
    const concisenessCard = {
      key: 'conciseness',
      title: 'Conciseness',
      primary: concisenessNum > 10
        ? `${formatPercentDisplay(concisenessNum)} excess`
        : 'Balanced delivery',
      secondary: concisenessNum > 10 ? null : `${formatPercentDisplay(Math.max(concisenessNum, 0))} excess`
    }
    if (concisenessNum > 10) {
      addCard(improvementHighlights, concisenessCard)
    } else {
      addCard(positiveHighlights, concisenessCard)
    }
  }

  if (topOpenerPctRaw !== null) {
    const sentenceStarterCard = {
      key: 'sentence-starters',
      title: 'Sentence Starters',
      primary: sentenceStarterPrimary,
      secondary: sentenceStarterSecondary
    }
    if (topOpenerPctRaw > 30) {
      addCard(improvementHighlights, sentenceStarterCard)
    } else {
      addCard(positiveHighlights, sentenceStarterCard)
    }
  }

  return (
    <div className="analytics-section">
      <section className="summary-block">
        <h3 className="section-title">
          <i className="fas fa-trophy" /> What went well
        </h3>
        <div className="summary-grid">
          {positiveHighlights.length > 0 ? (
            positiveHighlights.map((card) => renderSummaryCard(card, 'positive'))
          ) : (
            <div className="summary-empty">No standout positives captured yet.</div>
          )}
        </div>
      </section>

      <section className="summary-block">
        <h3 className="section-title">
          <i className="fas fa-lightbulb" /> What could have gone better
        </h3>
        <div className="summary-grid">
          {improvementHighlights.length > 0 ? (
            improvementHighlights.map((card) => renderSummaryCard(card, 'negative'))
          ) : (
            <div className="summary-empty">No major issues detected here.</div>
          )}
        </div>
      </section>

      <h3 className="section-title">
        <i className="fas fa-sliders-h" /> Detailed Breakdown
      </h3>

      {/* Filler Words */}
      <div className="metric-row">
        <div
          className="metric-header"
          onClick={() => handleToggle('filler', setExpandedFiller, expandedFiller)}
        >
          <span>Filler Words</span>
          <div className="metric-badge">
            {fillerPercentage}%
            <i className={`fas fa-chevron-${expandedFiller ? 'down' : 'right'}`} />
          </div>
        </div>
        {expandedFiller && (
          <div className="metric-content">
            {loadingState.filler ? (
              <MetricLoading />
            ) : (
              <div className={`feedback-box ${fillerPercentage < 4 ? 'success' : ''}`}>
                {fillerPercentage < 4 ? (
                  <>
                    <p>Well done! It's natural to have fewer than 4% fillers.</p>
                    {mumblingDetected && (
                      <p style={{ fontSize: '0.8rem', color: 'var(--accent-warning)', marginTop: '0.5rem' }}>
                        ⚠️ Some mumbling patterns detected - practice clear articulation
                      </p>
                    )}
                  </>
                ) : (
                  <>
                    <p>You used <strong>{fillerPercentage}%</strong> filler words. Try to reduce this to below 4%.</p>
                    {mumblingDetected && (
                      <p style={{ fontSize: '0.8rem', color: 'var(--accent-warning)', marginTop: '0.5rem' }}>
                        ⚠️ Mumbling patterns detected - practice clear articulation
                      </p>
                    )}
                  </>
                )}

                {Object.keys(fillerBreakdown).length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Filler words breakdown:</p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                      {Object.entries(fillerBreakdown)
                        .sort((a, b) => b[1] - a[1])
                        .map(([word, count]) => (
                          <span key={word} style={{
                            padding: '0.25rem 0.5rem',
                            background: 'rgba(139, 92, 246, 0.15)',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                            border: '1px solid rgba(139, 92, 246, 0.3)'
                          }}>
                            "{word}" ({count})
                          </span>
                        ))}
                    </div>
                  </div>
                )}

                {acousticCount > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>
                      Detected filler &amp; murmur events ({acousticCount}):
                    </p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                      {acousticEvents.slice(0, 4).map((event, idx) => {
                        const startSeconds = typeof event.start === 'number' ? event.start : null
                        const durationSeconds = typeof event.duration === 'number'
                          ? event.duration
                          : (typeof event.start === 'number' && typeof event.end === 'number'
                            ? Math.max(0, event.end - event.start)
                            : 0)
                        const confidence = typeof event.confidence === 'number'
                          ? event.confidence
                          : (typeof event.score === 'number' ? event.score : 0)
                        return (
                          <div
                            key={`${event.start}-${idx}`}
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              gap: '1rem',
                              background: 'rgba(248, 113, 113, 0.12)',
                              border: '1px solid rgba(248, 113, 113, 0.35)',
                              borderRadius: '6px',
                              padding: '0.45rem 0.6rem',
                              fontSize: '0.75rem',
                              color: 'var(--text-secondary)'
                            }}
                          >
                            <span>
                              {event.label || 'murmur'}{event.method ? ` (${event.method.replace('_', ' ')})` : ''} {startSeconds !== null && (
                                <>
                                  at <strong>{formatSeconds(startSeconds)}</strong>
                                </>
                              )}
                            </span>
                            <span>
                              {Math.round(confidence * 100)}% conf · {formatSeconds(durationSeconds)} long
                            </span>
                          </div>
                        )
                      })}
                      {acousticEvents.length > 4 && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                          +{acousticEvents.length - 4} more events detected
                        </span>
                      )}
                    </div>
                  </div>
                )}

                <a href="#" className="external-link" style={{ marginTop: '0.75rem', display: 'inline-block' }}>
                  How to avoid filler words <i className="fas fa-external-link-alt" />
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Weak Words */}
      <div className="metric-row">
        <div
          className="metric-header"
          onClick={() => handleToggle('weak', setExpandedWeak, expandedWeak)}
        >
          <span>Weak Words</span>
          <div className="metric-badge">
            {weakWordsPercentage}%
            <i className={`fas fa-chevron-${expandedWeak ? 'down' : 'right'}`} />
          </div>
        </div>
        {expandedWeak && (
          <div className="metric-content">
            {loadingState.weak ? (
              <MetricLoading />
            ) : (
              <div className="feedback-box">
                <p>You used <strong>{weakWordsPercentage}%</strong> weak words ({weakWordsFound.length} instances). Consider using stronger, more specific language.</p>

                {Object.keys(weakWordBreakdown).length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Weak words found:</p>
                    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                      {Object.entries(weakWordBreakdown).map(([word, count]) => {
                        const wordInfo = weakWordsFound.find(w => w.word.toLowerCase() === word.toLowerCase())
                        return (
                          <li key={word} style={{
                            padding: '0.5rem',
                            marginBottom: '0.375rem',
                            background: 'rgba(255, 193, 7, 0.1)',
                            borderRadius: '4px',
                            fontSize: '0.8rem'
                          }}>
                            <strong>"{word}"</strong> - used {count} time{count > 1 ? 's' : ''}
                            {wordInfo?.suggestion && (
                              <div style={{ marginTop: '0.25rem', color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                                💡 {wordInfo.suggestion}
                              </div>
                            )}
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )}

                <a
                  href="#"
                  className="external-link"
                  style={{ marginTop: '0.75rem', display: 'inline-block', cursor: 'pointer' }}
                  onClick={(e) => {
                    e.preventDefault()
                    setShowWeakWordsModal(true)
                  }}
                >
                  Learn about weak words <i className="fas fa-external-link-alt" />
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Sentence Openers */}
      <div className="metric-row">
        <div
          className="metric-header"
          onClick={() => handleToggle('sentence_openers', setExpandedSentenceOpeners, expandedSentenceOpeners)}
        >
          <span>Sentence Openers</span>
          <div className="metric-badge">
            {(() => {
              const sentenceOpenerAnalysis = reportData?.sentence_opener_analysis;
              if (!sentenceOpenerAnalysis || sentenceOpenerAnalysis.status === 'no_data') {
                return 'No data';
              }
              const varietyScore = sentenceOpenerAnalysis.variety_score || 100;
              return `${varietyScore}/100`;
            })()}
            <i className={`fas fa-chevron-${expandedSentenceOpeners ? 'down' : 'right'}`} />
          </div>
        </div>
        {expandedSentenceOpeners && (
          <div className="metric-content">
            {loadingState.sentence_openers ? (
              <MetricLoading />
            ) : (() => {
              const sentenceOpenerAnalysis = reportData?.sentence_opener_analysis;

              // No data case
              if (!sentenceOpenerAnalysis || sentenceOpenerAnalysis.status === 'no_data') {
                return (
                  <div className="feedback-box">
                    <p>Sentence opener variety looks balanced. Keep mixing how you begin sentences to maintain flow.</p>
                  </div>
                );
              }

              const {
                status,
                message,
                openers_found = {},
                recommendations = [],
                variety_score = 100
              } = sentenceOpenerAnalysis;

              const isGood = status === 'excellent' || status === 'good';

              return (
                <div className={`feedback-box ${isGood ? 'success' : ''}`}>
                  <p>{message}</p>

                  {Object.keys(openers_found).length > 0 && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Detected openers:</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        {Object.entries(openers_found)
                          .sort((a, b) => b[1].count - a[1].count)
                          .map(([opener, data]) => {
                            const isOverused = data.is_overused;
                            const severityColor = data.severity === 'high'
                              ? 'rgba(244, 67, 54, 0.25)'
                              : data.severity === 'medium'
                                ? 'rgba(255, 152, 0, 0.2)'
                                : 'rgba(139, 92, 246, 0.15)';
                            const borderColor = data.severity === 'high'
                              ? 'rgba(244, 67, 54, 0.4)'
                              : data.severity === 'medium'
                                ? 'rgba(255, 152, 0, 0.4)'
                                : 'rgba(139, 92, 246, 0.3)';

                            return (
                              <div
                                key={opener}
                                style={{
                                  padding: '0.5rem 0.7rem',
                                  background: severityColor,
                                  borderRadius: '6px',
                                  fontSize: '0.75rem',
                                  border: `1px solid ${borderColor}`,
                                  display: 'flex',
                                  flexDirection: 'column',
                                  gap: '0.25rem'
                                }}
                              >
                                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                  <strong>"{opener.toUpperCase()}"</strong>
                                  <span style={{ opacity: 0.8 }}>
                                    {data.count}x ({data.percentage}%)
                                  </span>
                                </div>
                                {isOverused && (
                                  <span style={{ fontSize: '0.7rem', color: 'var(--accent-warning)' }}>
                                    ⚠️ Overused
                                  </span>
                                )}
                              </div>
                            );
                          })}
                      </div>
                    </div>
                  )}

                  {recommendations.length > 0 && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>💡 Recommendations:</p>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {recommendations.map((rec, idx) => (
                          <div
                            key={idx}
                            style={{
                              padding: '0.5rem',
                              background: rec.severity === 'high'
                                ? 'rgba(244, 67, 54, 0.12)'
                                : 'rgba(255, 152, 0, 0.12)',
                              borderRadius: '6px',
                              fontSize: '0.75rem',
                              border: `1px solid ${rec.severity === 'high'
                                ? 'rgba(244, 67, 54, 0.3)'
                                : 'rgba(255, 152, 0, 0.3)'}`
                            }}
                          >
                            <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>
                              {rec.opener}: <span style={{ opacity: 0.8 }}>{rec.usage}</span>
                            </div>
                            <div style={{ fontSize: '0.7rem', opacity: 0.9 }}>
                              {rec.suggestion}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}
          </div>
        )}
      </div>

      {/* Vocabulary */}
      <div className="metric-row">
        <div
          className="metric-header"
          onClick={() => handleToggle('vocabulary', setExpandedVocabulary, expandedVocabulary)}
        >
          <span>Vocabulary</span>
          <div className={`metric-badge ${vocabularyStrength ? 'success' : ''}`}>
            {vocabularyRichnessPct} richness
            <i className={`fas fa-chevron-${expandedVocabulary ? 'down' : 'right'}`} />
          </div>
        </div>
        {expandedVocabulary && (
          <div className="metric-content">
            {loadingState.vocabulary ? (
              <MetricLoading />
            ) : (
              <div className={`feedback-box ${vocabularyStrength ? 'success' : ''}`}>
                {!vocabularyHasData && (
                  <p>Vocabulary metrics weren’t captured for this session.</p>
                )}

                {vocabularyHasData && (
                  <>
                    <p>
                      Lexical variety scored <strong>{vocabularyRichnessPct}%</strong> with <strong>{uniqueWords}</strong> unique words across {totalWordCount} total.
                    </p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      Complex word usage: <strong>{complexWords}</strong> ({complexWordPct}% of your vocabulary).
                    </p>
                  </>
                )}

                {advancedTerms.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>Advanced terms leaned on:</p>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                      {advancedTerms.slice(0, 6).map((term) => (
                        <span
                          key={term}
                          style={{
                            padding: '0.25rem 0.45rem',
                            background: 'rgba(59, 130, 246, 0.12)',
                            borderRadius: '4px',
                            fontSize: '0.72rem',
                            border: '1px solid rgba(59, 130, 246, 0.25)'
                          }}
                        >
                          {term}
                        </span>
                      ))}
                      {advancedTerms.length > 6 && (
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                          +{advancedTerms.length - 6} more
                        </span>
                      )}
                    </div>
                  </div>
                )}

                {displayVocabularyEnhancements.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '0.6rem', color: 'var(--text-primary)' }}>
                      Vocabulary Improvements Used
                    </p>
                    <div className="vocabulary-table-wrapper">
                      <table className="vocabulary-table">
                        <thead>
                          <tr>
                            <th>Original</th>
                            <th>Enhanced</th>
                            <th>Why</th>
                          </tr>
                        </thead>
                        <tbody>
                          {displayVocabularyEnhancements.slice(0, 6).map((enhancement, idx) => {
                            const originalText = (enhancement.context || enhancement.word || '').trim()
                            const enhancedText = Array.isArray(enhancement.suggestions) && enhancement.suggestions.length
                              ? enhancement.suggestions.slice(0, 3).join(', ')
                              : '—'
                            const reasonText = enhancement.reason || 'Improves clarity and tone.'
                            return (
                              <tr key={`${enhancement.word || 'enhancement'}-${idx}`}>
                                <td>{originalText || '—'}</td>
                                <td>{enhancedText}</td>
                                <td>{reasonText}</td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                    {displayVocabularyEnhancements.length > 6 && (
                      <p style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.4rem' }}>
                        +{displayVocabularyEnhancements.length - 6} additional substitutions available in the full report.
                      </p>
                    )}
                  </div>
                )}

                {vocabularySuggestionsList.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>Next steps:</p>
                    <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.8rem' }}>
                      {vocabularySuggestionsList.slice(0, 4).map((suggestion, idx) => (
                        <li key={idx} style={{ marginBottom: '0.35rem', color: 'var(--text-secondary)' }}>
                          {typeof suggestion === 'string' ? suggestion : suggestion?.advice || JSON.stringify(suggestion)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {displayVocabularyEnhancements.length === 0 && vocabularySuggestionsList.length === 0 && vocabularyMessage && (
                  <p style={{ fontStyle: 'italic', color: 'var(--text-secondary)' }}>{vocabularyMessage}</p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Conciseness */}
      <div className="metric-row">
        <div
          className="metric-header"
          onClick={() => handleToggle('conciseness', setExpandedConciseness, expandedConciseness)}
        >
          <span>Conciseness</span>
          <div className="metric-badge">
            {excessPercentage}% Excess
            <i className={`fas fa-chevron-${expandedConciseness ? 'down' : 'right'}`} />
          </div>
        </div>
        {expandedConciseness && (
          <div className="metric-content">
            {loadingState.conciseness ? (
              <MetricLoading />
            ) : (
              <div className="feedback-box">
                <p>Your speech has {excessPercentage}% excess. Try to be more concise and get to the point faster.</p>
                {reportData?.speaking_metrics?.total_duration && reportData?.speaking_metrics?.speaking_time && (
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                    You spoke for {reportData.speaking_metrics.speaking_time.toFixed(1)}s out of {reportData.speaking_metrics.total_duration.toFixed(1)}s total. Trim supporting detail to reclaim {(reportData.speaking_metrics.total_duration - reportData.speaking_metrics.speaking_time).toFixed(1)}s for key insights.
                  </p>
                )}
                <a
                  href="#"
                  className="external-link"
                  onClick={(e) => {
                    e.preventDefault()
                    alert('Tips for Concise Communication:\n\n1. Remove unnecessary filler words\n2. Use active voice instead of passive\n3. Get to the point in the first sentence\n4. Trim redundant adjectives and adverbs\n5. Replace long phrases with shorter alternatives\n6. Use bullet points for lists\n7. Avoid repetition - say it once clearly')
                  }}
                >
                  Tips for concise communication <i className="fas fa-external-link-alt" />
                </a>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ADVANCED Sentence Openers Analysis */}
      {(() => {
        const structureMetrics = reportData?.text_analysis?.structure_metrics || {}
        const openerAnalysis = structureMetrics.sentence_openers || {}
        const openerPercentages = openerAnalysis.opener_percentages || {}
        const totalSentences = openerAnalysis.total_sentences || 0
        const recommendations = openerAnalysis.recommendations || []

        // Get top weak openers (sorted by percentage)
        const topOpeners = Object.entries(openerPercentages)
          .filter(([_, pct]) => pct > 0)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)

        if (topOpeners.length > 0) {
          const topOpener = topOpeners[0]
          const openerName = topOpener[0].charAt(0).toUpperCase() + topOpener[0].slice(1)
          const openerPct = topOpener[1]

          return (
            <div className="metric-row">
              <div className="metric-header">
                <span>Sentence Openers</span>
                <div className="metric-badge warning">"{openerName}," {openerPct}%</div>
              </div>
              <div className="metric-content">
                <div className="feedback-box">
                  <p>
                    <strong>{openerPct}%</strong> of sentences start with "{openerName}".
                    {openerPct > 20 && " This creates repetitive patterns. Vary your sentence openers for better engagement."}
                  </p>

                  {topOpeners.length > 1 && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Other weak openers detected:</p>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                        {topOpeners.slice(1).map(([opener, pct]) => (
                          <span key={opener} style={{
                            padding: '0.25rem 0.5rem',
                            background: 'rgba(245, 158, 11, 0.15)',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                            border: '1px solid rgba(245, 158, 11, 0.3)'
                          }}>
                            "{opener}" ({pct}%)
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {recommendations.length > 0 && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Recommendations:</p>
                      <ul style={{ margin: 0, paddingLeft: '1.25rem', fontSize: '0.8rem' }}>
                        {recommendations.map((rec, idx) => (
                          <li key={idx} style={{ marginBottom: '0.25rem', color: 'var(--text-secondary)' }}>
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <a href="#" className="external-link" style={{ marginTop: '0.75rem', display: 'inline-block' }}>
                    Learn about sentence variety <i className="fas fa-external-link-alt" />
                  </a>
                </div>
              </div>
            </div>
          )
        }

        return (
          <div className="metric-row">
            <div className="metric-content">
              <div className="feedback-box" style={{ color: 'var(--text-primary)', padding: '1rem' }}>
                <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                  Great job! No major issues detected in this area.
                </p>
              </div>
            </div>
          </div>
        )
      })()}
    </div>
  )
}

const DeliveryAnalytics = ({ speakingRate, pauseCount, eyeContactPercent, pauseSummary, pausesDetailed, isAudioOnly = false }) => {
  const [expandedPacing, setExpandedPacing] = useState(true)
  const [expandedEyeContact, setExpandedEyeContact] = useState(false)
  const [expandedPauses, setExpandedPauses] = useState(false)
  const [expandedPausesList, setExpandedPausesList] = useState(false) // SEPARATE state for list expansion
  const [loadingState, setLoadingState] = useState({
    pacing: false,
    eye: false,
    pause: false
  })

  const hasSpeakingRate = typeof speakingRate === 'number' && !Number.isNaN(speakingRate)
  const pacingPositive = hasSpeakingRate && speakingRate >= 135 && speakingRate <= 170
  const pacingSlow = hasSpeakingRate && speakingRate < 135
  const pacingFast = hasSpeakingRate && speakingRate > 170

  const hasEyeContact = typeof eyeContactPercent === 'number' && !Number.isNaN(eyeContactPercent)
  const eyeContactPositive = hasEyeContact && eyeContactPercent >= 55

  const pausesList = Array.isArray(pausesDetailed) ? pausesDetailed : []
  const totalPauses = typeof pauseCount === 'number' ? pauseCount : pausesList.length
  const totalPauseTime = typeof pauseSummary.total_pause_time === 'number' ? pauseSummary.total_pause_time : null
  const longestPause = typeof pauseSummary.longest_pause === 'number' ? pauseSummary.longest_pause : null
  const pausePositive = totalPauses >= 2 && totalPauses <= 12

  const positiveMetrics = [
    pacingPositive ? 'pacing' : null,
    (!isAudioOnly && eyeContactPositive) ? 'eyeContact' : null,
    pausePositive ? 'pauses' : null
  ].filter(Boolean)

  const improvementMetrics = [
    hasSpeakingRate && !pacingPositive ? 'pacing' : null,
    (!isAudioOnly && hasEyeContact && !eyeContactPositive) ? 'eyeContact' : null,
    (!pausePositive && typeof totalPauses === 'number') ? 'pauses' : null
  ].filter(Boolean)

  const formatSeconds = (seconds) => {
    if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '--'
    return `${seconds.toFixed(1)}s`
  }

  const triggerLoading = useCallback((sectionKey, shouldOpen) => {
    if (shouldOpen) {
      setLoadingState((prev) => ({ ...prev, [sectionKey]: true }))
      setTimeout(() => {
        setLoadingState((prev) => ({ ...prev, [sectionKey]: false }))
      }, 320)
    }
  }, [])

  const handleToggle = (sectionKey, setter, currentValue) => {
    const nextValue = !currentValue
    setter(nextValue)
    triggerLoading(sectionKey, nextValue)
  }

  const renderPacingCard = () => (
    <div className="metric-row">
      <div
        className="metric-header"
        onClick={() => handleToggle('pacing', setExpandedPacing, expandedPacing)}
      >
        <span>Pacing</span>
        <div className="metric-badge">
          {hasSpeakingRate ? `${speakingRate.toFixed(0)} WPM` : 'No data'}
          <i className={`fas fa-chevron-${expandedPacing ? 'down' : 'right'}`} />
        </div>
      </div>
      {expandedPacing && (
        <div className="metric-content">
          {loadingState.pacing ? (
            <MetricLoading />
          ) : (
            <>
              <div className={`feedback-box ${pacingPositive ? 'success' : ''}`}>
                {!hasSpeakingRate && (
                  <p>Pacing metrics weren’t available for this session.</p>
                )}
                {hasSpeakingRate && pacingPositive && (
                  <p>You spoke at {speakingRate.toFixed(0)} WPM, right inside the conversational range (135–170).</p>
                )}
                {hasSpeakingRate && pacingSlow && (
                  <p>You averaged {speakingRate.toFixed(0)} WPM, which feels slow. Trim filler phrases and rehearse with a metronome track to reach 140–160 WPM.</p>
                )}
                {hasSpeakingRate && pacingFast && (
                  <p>You averaged {speakingRate.toFixed(0)} WPM, which is fast. Add breath-led pauses and emphasise keywords to stay below 170 WPM.</p>
                )}
              </div>
              {hasSpeakingRate && (
                <div className="pacing-visualization">
                  <div className="pace-gauge">
                    <div className="gauge-label-left">Slow</div>
                    <div className="gauge-label-center">Conversational</div>
                    <div className="gauge-label-right">Quick</div>
                    <div className="gauge-bar">
                      <div
                        className="gauge-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, ((speakingRate - 60) / 180) * 100))}%`
                        }}
                      />
                      <div
                        className="gauge-indicator"
                        style={{
                          left: `${Math.min(100, Math.max(0, ((speakingRate - 60) / 180) * 100))}%`
                        }}
                      />
                    </div>
                    <div className="gauge-value">{speakingRate.toFixed(0)} WPM</div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )

  const renderEyeContactCard = () => (
    <div className="metric-row">
      <div
        className="metric-header"
        onClick={() => handleToggle('eye', setExpandedEyeContact, expandedEyeContact)}
      >
        <span>Eye Contact</span>
        <div className={`metric-badge ${eyeContactPositive ? 'success' : 'warning'}`}>
          {hasEyeContact ? `~${eyeContactPercent.toFixed(0)}%` : 'No data'}
          <i className={`fas fa-chevron-${expandedEyeContact ? 'down' : 'right'}`} />
        </div>
      </div>
      {expandedEyeContact && (
        <div className="metric-content">
          {loadingState.eye ? (
            <MetricLoading />
          ) : (
            <div className="feedback-box">
              {!hasEyeContact && (
                <p>Eye-contact data was not captured this session.</p>
              )}
              {hasEyeContact && eyeContactPositive && (
                <p>You maintained direct gaze for roughly {eyeContactPercent.toFixed(0)}% of the timeline. Keep rotating between left, centre and right viewers to hold engagement.</p>
              )}
              {hasEyeContact && !eyeContactPositive && (
                <p>Eye contact held for about {eyeContactPercent.toFixed(0)}%. Practice the triangle gaze pattern (left–centre–right) and anchor your notes closer to the camera.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )

  const renderPauseCard = () => (
    <div className="metric-row">
      <div
        className="metric-header"
        onClick={() => handleToggle('pause', setExpandedPauses, expandedPauses)}
      >
        <span>Pauses</span>
        <div className={`metric-badge ${pausePositive ? 'success' : ''}`}>
          {typeof totalPauses === 'number' ? `${totalPauses} pause${totalPauses === 1 ? '' : 's'}` : 'No data'}
          <i className={`fas fa-chevron-${expandedPauses ? 'down' : 'right'}`} />
        </div>
      </div>
      {expandedPauses && (
        <div className="metric-content">
          {loadingState.pause ? (
            <MetricLoading />
          ) : (
            <div className="feedback-box">
              {typeof totalPauses !== 'number' && (
                <p>Pause analytics were not recorded for this session.</p>
              )}
              {typeof totalPauses === 'number' && pausePositive && (
                <p>
                  You used {totalPauses} pauses{totalPauseTime !== null && ` (total ${formatSeconds(totalPauseTime)})`}
                  {longestPause !== null && `, longest ${formatSeconds(longestPause)}`}. This keeps phrasing natural.
                </p>
              )}
              {typeof totalPauses === 'number' && !pausePositive && (
                <>
                  {totalPauses === 0 ? (
                    <p>No pauses detected. Insert breath breaks at key transitions to let your message land.</p>
                  ) : (
                    <p>
                      {totalPauses} pauses detected{longestPause !== null && `; the longest stretched to ${formatSeconds(longestPause)}`}. Tighten transitions and rehearse with a beat timer to reduce filler silences.
                    </p>
                  )}
                  {pausesList.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      <ul style={{ marginTop: '0.5rem', paddingLeft: '1.25rem', fontSize: '0.7rem', color: 'var(--text-secondary)' }}>
                        {pausesList.slice(0, expandedPausesList ? pausesList.length : 4).map((pause, idx) => {
                          const pauseStart = typeof pause.start === 'number' ? formatSeconds(pause.start) : '--'
                          const pauseDuration = typeof pause.duration === 'number' ? formatSeconds(pause.duration) : '--'
                          return (
                            <li key={idx}>
                              Pause at {pauseStart} lasting {pauseDuration}
                            </li>
                          )
                        })}
                      </ul>
                      {pausesList.length > 4 && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            setExpandedPausesList(!expandedPausesList) // FIX: Use separate state
                          }}
                          style={{
                            marginTop: '0.5rem',
                            padding: '0.25rem 0.5rem',
                            fontSize: '0.7rem',
                            background: 'transparent',
                            border: '1px solid var(--border-color)',
                            borderRadius: '4px',
                            color: 'var(--text-secondary)',
                            cursor: 'pointer'
                          }}
                        >
                          {expandedPausesList ? 'Show less' : `+${pausesList.length - 4} more pauses`}
                        </button>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )

  return (
    <div className="analytics-section">
      {positiveMetrics.length > 0 && (
        <>
          <h3 className="section-title">
            <i className="fas fa-trophy" /> What went well
          </h3>
          {pacingPositive && renderPacingCard()}
          {!isAudioOnly && eyeContactPositive && renderEyeContactCard()}
          {pausePositive && renderPauseCard()}
        </>
      )}

      {improvementMetrics.length > 0 && (
        <>
          <h3 className="section-title">
            <i className="fas fa-lightbulb" /> What could have gone better
          </h3>
          {!pacingPositive && hasSpeakingRate && renderPacingCard()}
          {!isAudioOnly && !eyeContactPositive && hasEyeContact && renderEyeContactCard()}
          {!pausePositive && typeof totalPauses === 'number' && renderPauseCard()}
        </>
      )}

      {positiveMetrics.length === 0 && improvementMetrics.length === 0 && (
        <div className="metric-row">
          <div className="metric-content">
            <div className="feedback-box">
              <p style={{ color: 'var(--text-secondary)' }}>
                Delivery metrics were not captured for this session.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const AdvancedAnalytics = ({
  audioAnalytics = {},
  visualAnalytics = {},
  textAnalytics = {},
  structureMetrics = {},
  contentMetrics = {},
  voiceConfidence = null,
  transcript = '',
  isAudioOnly = false
}) => {
  const numberOrNull = (value) => {
    if (value === null || value === undefined) return null
    const num = Number(value)
    return Number.isFinite(num) ? num : null
  }

  const fillerTrend = audioAnalytics.filler_trend || {}
  // Ensure pause_cadence has proper structure with defaults
  const rawPauseCadence = audioAnalytics.pause_cadence || {}
  const pauseCadence = {
    counts: rawPauseCadence.counts || { short: 0, medium: 0, long: 0 },
    durations: rawPauseCadence.durations || { short: 0.0, medium: 0.0, long: 0.0 },
    average_duration: rawPauseCadence.average_duration || 0.0,
    total_pause_time: rawPauseCadence.total_pause_time || 0.0
  }
  const openingConfidence = audioAnalytics.opening_confidence || {}
  const totalSentences = numberOrNull(structureMetrics.total_sentences) || 0

  const tensionSummary = visualAnalytics.tension_summary || {}
  const emotionTimelineSmoothed = Array.isArray(visualAnalytics.emotion_timeline_smoothed)
    ? visualAnalytics.emotion_timeline_smoothed
    : []

  let topicCoherenceScore = numberOrNull(
    textAnalytics.topic_coherence_score ??
    textAnalytics.topic_coherence ??
    contentMetrics.content_coherence
  )
  let topTopics = Array.isArray(textAnalytics.top_topics) ? [...textAnalytics.top_topics] : []
  let keywordCoverage = textAnalytics.keyword_coverage || {}
  let sentencePatternScore = numberOrNull(
    textAnalytics.sentence_pattern_score ??
    structureMetrics.sentence_pattern_score ??
    structureMetrics.opener_diversity
  )
  const repetitionAlerts = Array.isArray(textAnalytics.repetition_alerts)
    ? textAnalytics.repetition_alerts
    : []
  const keywordDetails = Array.isArray(textAnalytics.keyword_details) ? textAnalytics.keyword_details : []
  let sentencePatternBreakdown = textAnalytics.sentence_pattern_breakdown || {}

  const fallbackSentencePattern = useMemo(() => {
    if (typeof transcript !== 'string' || transcript.trim().length < 30) {
      return null
    }
    const rawSentences = transcript
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter(Boolean)
    if (!rawSentences.length) {
      return null
    }

    const lengths = rawSentences.map((sentence) => {
      const tokens = sentence.split(/\s+/).filter(Boolean)
      return tokens.length
    })
    if (!lengths.length) {
      return null
    }

    const avgLength = lengths.reduce((sum, len) => sum + len, 0) / lengths.length
    const variance = lengths.reduce((sum, len) => sum + (len - avgLength) ** 2, 0) / lengths.length
    const lengthStd = Math.sqrt(variance)
    const shortPct = (lengths.filter((len) => len <= 8).length / lengths.length) * 100
    const longPct = (lengths.filter((len) => len >= 25).length / lengths.length) * 100
    const varietyPenalty = Math.min(40, Math.abs(avgLength - 18) * 2) + Math.min(25, lengthStd * 1.5)
    const pacingPenalty = Math.min(15, shortPct * 0.35) + Math.min(15, longPct * 0.4)
    const balanceBonus = avgLength >= 10 && avgLength <= 20 ? 6 : 0
    const score = Math.round(Math.max(28, Math.min(94, 94 - varietyPenalty - pacingPenalty + balanceBonus)))

    return {
      score,
      breakdown: {
        average_length: Number(avgLength.toFixed(1)),
        length_std: Number(lengthStd.toFixed(1)),
        short_pct: Number(shortPct.toFixed(1)),
        long_pct: Number(longPct.toFixed(1))
      }
    }
  }, [transcript])

  const formatSeconds = (seconds) => {
    if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '--'
    return `${seconds.toFixed(1)}s`
  }

  const trendBuckets = Array.isArray(fillerTrend.trend) ? fillerTrend.trend : []
  const topFillers = Array.isArray(fillerTrend.top_labels) ? fillerTrend.top_labels : []

  // Fallback: Count fillers directly from transcript if trend data is missing
  const fallbackFillerCount = useMemo(() => {
    if (trendBuckets.length > 0 || !transcript || typeof transcript !== 'string') {
      return null
    }

    const fillerWords = ['uh', 'um', 'ah', 'er', 'hmm', 'erm', 'eh', 'umm', 'uhh', 'ahh', 'ehh', 'err', 'uhm', 'ahm', 'ummm', 'uhhh', 'ahhh', 'ehhh', 'errr', 'oh']
    const words = transcript.toLowerCase().split(/\s+/)
    const fillerCounts = {}
    let totalFillers = 0

    words.forEach(word => {
      const cleanWord = word.replace(/[.,!?;:]/g, '')
      if (fillerWords.includes(cleanWord)) {
        fillerCounts[cleanWord] = (fillerCounts[cleanWord] || 0) + 1
        totalFillers++
      }
    })

    const topFillersList = Object.entries(fillerCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)

    return {
      total: totalFillers,
      breakdown: topFillersList
    }
  }, [transcript, trendBuckets.length])

  const pauseCounts = pauseCadence.counts || {}
  const pauseDurations = pauseCadence.durations || {}
  let confidenceScore = numberOrNull(openingConfidence.opening_confidence)
  let confidenceSource = confidenceScore !== null ? 'measured' : 'estimated'
  if (confidenceScore === null) {
    const openingFillerCount = numberOrNull(openingConfidence.opening_filler_count)
    const openingPauseCount = numberOrNull(openingConfidence.opening_pause_count)
    let estimate = 88
    if (openingFillerCount !== null) {
      estimate -= Math.max(0, openingFillerCount) * 6
    }
    if (openingPauseCount !== null) {
      estimate -= Math.max(0, openingPauseCount - 1) * 4
    }
    const voiceConfidenceNumber = numberOrNull(voiceConfidence)
    if (voiceConfidenceNumber !== null) {
      estimate = (estimate * 0.4) + (voiceConfidenceNumber * 0.6)
    }
    confidenceScore = Math.max(0, Math.min(100, estimate))
  }

  if ((!topTopics || topTopics.length === 0) && Array.isArray(contentMetrics.key_topics)) {
    keywordCoverage = {
      total_keywords: contentMetrics.key_topics.length,
      top_keywords: contentMetrics.key_topics.slice(0, 6),
      keyword_density: numberOrNull(contentMetrics.keyword_density) ?? 0,
      coverage_ratio: totalSentences
        ? Math.min(1, contentMetrics.key_topics.length / totalSentences)
        : 0
    }
    topTopics = contentMetrics.key_topics.slice(0, 6)
  }
  if ((!keywordCoverage || typeof keywordCoverage !== 'object') && contentMetrics.keyword_counts) {
    const keywordCounts = contentMetrics.keyword_counts || {}
    const totals = Object.values(keywordCounts).reduce((sum, count) => sum + Number(count || 0), 0)
    const fallbackTopics = Object.keys(keywordCounts)
    keywordCoverage = {
      total_keywords: totals,
      top_keywords: fallbackTopics.slice(0, 6),
      keyword_density: totals && totalSentences
        ? Number(((totals / totalSentences) * 100).toFixed(2))
        : 0,
      coverage_ratio: totalSentences
        ? Math.min(1, fallbackTopics.length / totalSentences)
        : 0
    }
    if (!topTopics || topTopics.length === 0) {
      topTopics = fallbackTopics.slice(0, 6)
    }
  }

  if (topicCoherenceScore === null) {
    const coverageRatio = keywordCoverage?.total_keywords && totalSentences
      ? Math.min(1, keywordCoverage.total_keywords / Math.max(totalSentences, 1))
      : 0
    const topicSpan = (Array.isArray(topTopics) ? new Set(topTopics).size : 0) || (keywordCoverage.top_keywords?.length || 0)
    const diversity = numberOrNull(structureMetrics.opener_diversity) ?? 0.4
    const base = (coverageRatio * 60) + (Math.min(1, diversity + 0.25) * 25) + Math.min(25, topicSpan * 3.5)
    topicCoherenceScore = Math.round(Math.max(18, Math.min(90, base)))
  }

  if (sentencePatternScore === null) {
    const variance = numberOrNull(structureMetrics.sentence_variety)
    const diversity = numberOrNull(structureMetrics.opener_diversity)
    if (variance !== null || diversity !== null) {
      const normalizedVar = variance !== null ? Math.max(0, Math.min(1, variance / 10)) : 0.5
      const normalizedDiv = diversity !== null ? Math.max(0, Math.min(1, diversity)) : 0.4
      sentencePatternScore = Math.round(((1 - Math.abs(normalizedVar - 0.4)) * 50 + normalizedDiv * 50))
    } else {
      sentencePatternScore = 0
    }
  }

  if ((sentencePatternScore === null || sentencePatternScore <= 0) && fallbackSentencePattern) {
    sentencePatternScore = fallbackSentencePattern.score
    if (!sentencePatternBreakdown || Object.keys(sentencePatternBreakdown).length === 0) {
      sentencePatternBreakdown = fallbackSentencePattern.breakdown
    }
  }

  if (!keywordCoverage || typeof keywordCoverage !== 'object') {
    keywordCoverage = { total_keywords: 0, keyword_density: 0, top_keywords: [], coverage_ratio: 0 }
  } else {
    keywordCoverage = {
      total_keywords: keywordCoverage.total_keywords ?? 0,
      keyword_density: numberOrNull(keywordCoverage.keyword_density) ?? 0,
      top_keywords: Array.isArray(keywordCoverage.top_keywords) ? keywordCoverage.top_keywords : [],
      coverage_ratio: numberOrNull(keywordCoverage.coverage_ratio) ?? (
        keywordCoverage.total_keywords && totalSentences
          ? Math.min(1, keywordCoverage.total_keywords / totalSentences)
          : 0
      )
    }
  }

  if (!keywordCoverage.total_keywords && Array.isArray(keywordDetails) && keywordDetails.length) {
    keywordCoverage.total_keywords = keywordDetails.reduce((sum, item) => sum + Number(item.count || 0), 0)
    keywordCoverage.top_keywords = keywordDetails.slice(0, 6).map((item) => item.word)
    if (!keywordCoverage.keyword_density && totalSentences > 0) {
      keywordCoverage.keyword_density = Number(((keywordCoverage.total_keywords / totalSentences) * 100).toFixed(2))
    }
    if (!keywordCoverage.coverage_ratio && totalSentences > 0) {
      keywordCoverage.coverage_ratio = Math.min(1, keywordCoverage.top_keywords.length / totalSentences)
    }
  }

  const [expandedEmotionTimeline, setExpandedEmotionTimeline] = useState(false)

  const renderTimelineSnippet = () => {
    if (!emotionTimelineSmoothed.length) {
      return <p className="muted">No emotion timeline available.</p>
    }

    const displayCount = expandedEmotionTimeline ? emotionTimelineSmoothed.length : 5
    const snippet = emotionTimelineSmoothed.slice(0, displayCount)
    return (
      <div>
        <ul className="trend-list">
          {snippet.map((entry, idx) => {
            // DEFENSIVE: Handle both probability (0-1) and percentage (0-100) formats
            const rawConfidence = entry.confidence
            let percentageValue = '--'

            if (typeof rawConfidence === 'number' && !isNaN(rawConfidence)) {
              // If confidence > 1, it's already a percentage (backend bug)
              // If confidence <= 1, it's a probability that needs * 100
              if (rawConfidence > 1) {
                percentageValue = `${Math.round(rawConfidence)}%`
              } else {
                percentageValue = `${Math.round(rawConfidence * 100)}%`
              }
            }

            return (
              <li key={`${entry.timestamp}-${idx}`}>
                <span>{formatSeconds(entry.timestamp || 0)}</span>
                <span>{entry.dominant_emotion || 'neutral'}</span>
                <span>{percentageValue}</span>
              </li>
            )
          })}
        </ul>
        {emotionTimelineSmoothed.length > 5 && (
          <button
            onClick={() => setExpandedEmotionTimeline(!expandedEmotionTimeline)}
            style={{
              marginTop: '0.5rem',
              padding: '0.25rem 0.5rem',
              fontSize: '0.7rem',
              background: 'transparent',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              color: 'var(--text-secondary)',
              cursor: 'pointer'
            }}
          >
            {expandedEmotionTimeline ? 'Show less' : `+${emotionTimelineSmoothed.length - 5} more entries`}
          </button>
        )}
      </div>
    )
  }

  return (
    <div className="advanced-analytics">
      <div className="advanced-section">
        <h3 className="section-title"><i className="fas fa-wave-square" /> Audio Intelligence</h3>
        <div className="advanced-grid">
          <div className="advanced-card">
            <div className="card-title">Filler Trend</div>
            {trendBuckets.length === 0 && !fallbackFillerCount ? (
              <p className="muted">No filler data available.</p>
            ) : trendBuckets.length === 0 && fallbackFillerCount ? (
              <div>
                <div style={{ padding: '0.75rem', background: 'rgba(244, 67, 54, 0.1)', borderRadius: '6px', marginBottom: '0.75rem' }}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--accent-warning)', marginBottom: '0.5rem' }}>
                    {fallbackFillerCount.total} Fillers Detected
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', margin: 0 }}>
                    Filler words significantly impact your delivery. Try to pause instead of using "uh" or "um".
                  </p>
                </div>
                {fallbackFillerCount.breakdown.length > 0 && (
                  <div className="trend-summary">
                    <span className="muted" style={{ fontSize: '0.8rem', fontWeight: 600 }}>Breakdown:</span>
                    <div className="chip-row" style={{ marginTop: '0.5rem' }}>
                      {fallbackFillerCount.breakdown.map(([label, count]) => (
                        <span key={label} className="trend-chip emphasis" style={{
                          padding: '0.4rem 0.6rem',
                          background: 'rgba(244, 67, 54, 0.15)',
                          border: '1px solid rgba(244, 67, 54, 0.3)',
                          borderRadius: '4px',
                          fontSize: '0.75rem',
                          fontWeight: 600
                        }}>
                          "{label}" × {count}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div style={{ marginTop: '0.75rem', padding: '0.5rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '6px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-primary)' }}>
                    <strong>💡 Quick Fix:</strong>
                    <div style={{ marginTop: '0.25rem', color: 'var(--text-secondary)' }}>
                      Practice pausing for 1-2 seconds instead of saying filler words. Record yourself and count improvements.
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <>
                <ul className="trend-list">
                  {trendBuckets.slice(0, 4).map((bucket, idx) => (
                    <li key={`${bucket.bucket_start}-${idx}`}>
                      <span>{bucket.bucket_label}</span>
                      <span>
                        {Object.entries(bucket.counts || {}).map(([label, count]) => (
                          <span key={label} className="trend-chip">
                            {label}: {count}
                          </span>
                        ))}
                      </span>
                    </li>
                  ))}
                </ul>
                {topFillers.length > 0 && (
                  <div className="trend-summary">
                    <span className="muted">Top fillers:</span>
                    <div className="chip-row">
                      {topFillers.slice(0, 5).map(([label, count]) => (
                        <span key={label} className="trend-chip emphasis">
                          {label} ({count})
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          <div className="advanced-card">
            <div className="card-title">Pause Cadence</div>
            <div className="pause-grid">
              {['short', 'medium', 'long'].map((bucket) => (
                <div key={bucket} className="pause-cell">
                  <div className="pause-label">{bucket.toUpperCase()}</div>
                  <div className="pause-count">{pauseCounts[bucket] || 0}</div>
                  <div className="pause-duration">
                    {(pauseDurations[bucket] || 0).toFixed(1)}s
                  </div>
                </div>
              ))}
            </div>
            <p className="muted">
              Avg pause length:{' '}
              {(pauseCadence.average_duration || 0).toFixed(2)}s
            </p>
          </div>

          <div className="advanced-card">
            <div className="card-title">Opening Confidence</div>
            {confidenceScore !== null ? (
              <div className="gauge-card">
                <div className="gauge-bar">
                  <div
                    className="gauge-fill"
                    style={{ width: `${Math.min(100, Math.max(0, confidenceScore))}%` }}
                  />
                </div>
                <div className="gauge-value">{confidenceScore.toFixed(1)} / 100</div>
                <div className="muted">
                  Filler hits: {openingConfidence.opening_filler_count || 0} · Pauses:{' '}
                  {openingConfidence.opening_pause_count || 0}
                </div>
                {confidenceSource === 'estimated' && (
                  <div className="muted" style={{ fontSize: '0.65rem' }}>
                    Estimated from filler and pause trends.
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">Opening confidence score not available.</p>
            )}
          </div>
        </div>
      </div>

      {/* Only show Visual Presence for video files */}
      {!isAudioOnly && (
        <div className="advanced-section">
          <h3 className="section-title"><i className="fas fa-user-circle" /> Visual Presence</h3>
          <div className="advanced-grid">
            <div className="advanced-card">
              <div className="card-title">Tension Ratio</div>
              {typeof tensionSummary.tension_percentage === 'number' ? (
                <div className="gauge-card">
                  <div className="gauge-bar">
                    <div
                      className="gauge-fill danger"
                      style={{ width: `${Math.min(100, Math.max(0, tensionSummary.tension_percentage))}%` }}
                    />
                  </div>
                  <div className="gauge-value">{tensionSummary.tension_percentage.toFixed(1)}%</div>
                  <div className="muted">
                    Eye-contact stability:{' '}
                    {typeof tensionSummary.eye_contact_stability === 'number'
                      ? `${tensionSummary.eye_contact_stability.toFixed(0)}%`
                      : '--'}
                  </div>
                  <div className="muted">
                    Avg eye contact:{' '}
                    {typeof tensionSummary.avg_eye_contact_pct === 'number'
                      ? `${tensionSummary.avg_eye_contact_pct.toFixed(0)}%`
                      : '--'}
                  </div>
                </div>
              ) : (
                <p className="muted">Tension analytics not available.</p>
              )}
            </div>

            <div className="advanced-card">
              <div className="card-title">Emotion Timeline (smoothed)</div>
              {renderTimelineSnippet()}
            </div>
          </div>
        </div>
      )}

      <div className="advanced-section">
        <h3 className="section-title"><i className="fas fa-font" /> Narrative Cohesion</h3>
        <div className="advanced-grid">
          <div className="advanced-card">
            <div className="card-title">Topic Coherence</div>
            {topicCoherenceScore !== null ? (
              <div className="gauge-card">
                <div className="gauge-bar">
                  <div
                    className="gauge-fill"
                    style={{ width: `${Math.min(100, Math.max(0, topicCoherenceScore))}%` }}
                  />
                </div>
                <div className="gauge-value">{topicCoherenceScore.toFixed(1)} / 100</div>
                {topTopics.length > 0 && (
                  <div className="chip-row">
                    {topTopics.slice(0, 4).map((topic) => (
                      <span key={topic} className="trend-chip">
                        {topic}
                      </span>
                    ))}
                  </div>
                )}
                {keywordDetails.length === 0 && (
                  <p className="muted" style={{ marginTop: '0.45rem' }}>
                    Limited topical phrases detected—add concrete details to boost coherence.
                  </p>
                )}
                {keywordDetails.length > 0 && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <button
                      onClick={() => {
                        const section = document.getElementById('topic-coherence-examples')
                        if (section) {
                          section.style.display = section.style.display === 'none' ? 'block' : 'none'
                        }
                      }}
                      style={{
                        padding: '0.4rem 0.75rem',
                        fontSize: '0.75rem',
                        background: 'var(--accent-primary)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => e.target.style.opacity = '0.85'}
                      onMouseLeave={(e) => e.target.style.opacity = '1'}
                    >
                      <i className="fas fa-eye" style={{ marginRight: '0.4rem' }} />
                      Show Examples
                    </button>
                    <div id="topic-coherence-examples" style={{ display: 'none', marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
                        Detected Keywords & Phrases:
                      </p>
                      <ul className="trend-list compact">
                        {keywordDetails.slice(0, 8).map((detail, idx) => (
                          <li key={`${detail.word}-${idx}`} style={{ marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                              <div style={{ flex: 1 }}>
                                <strong style={{ color: 'var(--accent-primary)' }}>{detail.word}</strong>
                                <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem', opacity: 0.7 }}>
                                  · {detail.count} mention{detail.count === 1 ? '' : 's'}
                                </span>
                                {detail.example && (
                                  <div style={{ marginTop: '0.25rem', fontSize: '0.7rem', fontStyle: 'italic', color: 'var(--text-secondary)' }}>
                                    "{detail.example}"
                                  </div>
                                )}
                              </div>
                            </div>
                          </li>
                        ))}
                      </ul>
                      <button
                        onClick={() => {
                          document.getElementById('topic-coherence-examples').style.display = 'none'
                        }}
                        style={{
                          marginTop: '0.5rem',
                          padding: '0.3rem 0.6rem',
                          fontSize: '0.7rem',
                          background: 'transparent',
                          color: 'var(--text-secondary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '4px',
                          cursor: 'pointer'
                        }}
                      >
                        Hide Examples
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">Topic coherence score unavailable.</p>
            )}
          </div>

          <div className="advanced-card">
            <div className="card-title">Keyword Coverage</div>
            {keywordCoverage.total_keywords > 0 ? (
              <div className="keyword-summary">
                <p>Total keywords: {keywordCoverage.total_keywords}</p>
                {typeof keywordCoverage.keyword_density === 'number' && (
                  <p>Density: {keywordCoverage.keyword_density.toFixed(2)}%</p>
                )}
                {typeof keywordCoverage.coverage_ratio === 'number' && keywordCoverage.coverage_ratio > 0 && (
                  <p>
                    Coverage: {(keywordCoverage.coverage_ratio * 100).toFixed(0)}% of sentences surfaced key phrases
                  </p>
                )}
                {Array.isArray(keywordCoverage.top_keywords) && keywordCoverage.top_keywords.length > 0 && (
                  <div className="chip-row">
                    {keywordCoverage.top_keywords.slice(0, 6).map((keyword) => (
                      <span key={keyword} className="trend-chip subtle">
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
                {keywordDetails.length > 0 && (
                  <div style={{ marginTop: '0.45rem', fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                    <p style={{ marginBottom: '0.3rem', fontWeight: 600 }}>Most cited phrases:</p>
                    <ul style={{ margin: 0, paddingLeft: '1.1rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      {keywordDetails.slice(0, 4).map((detail, idx) => (
                        <li key={`${detail.word}-detail-${idx}`}>
                          <strong>{detail.word}</strong> used {detail.count} time{detail.count === 1 ? '' : 's'}{detail.example && ` — “${detail.example}”`}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">Keyword coverage data unavailable.</p>
            )}
          </div>

          <div className="advanced-card">
            <div className="card-title">Sentence Pattern Score</div>
            {sentencePatternScore !== null ? (
              <div className="gauge-card">
                <div className="gauge-bar">
                  <div
                    className="gauge-fill"
                    style={{ width: `${Math.min(100, Math.max(0, sentencePatternScore))}%` }}
                  />
                </div>
                <div className="gauge-value">{sentencePatternScore.toFixed(1)} / 100</div>
                {sentencePatternBreakdown && (
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                      <div style={{ marginBottom: '0.25rem' }}>
                        <strong>Avg length:</strong> {sentencePatternBreakdown.average_length || '—'} words
                      </div>
                      <div style={{ marginBottom: '0.25rem' }}>
                        <strong>Std dev:</strong> {sentencePatternBreakdown.length_std || '—'}
                      </div>
                      <div style={{ marginBottom: '0.25rem' }}>
                        <strong>Short (≤8 words):</strong> {sentencePatternBreakdown.short_pct || 0}%
                      </div>
                      <div>
                        <strong>Long (≥25 words):</strong> {sentencePatternBreakdown.long_pct || 0}%
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        const section = document.getElementById('sentence-pattern-examples')
                        if (section) {
                          section.style.display = section.style.display === 'none' ? 'block' : 'none'
                        }
                      }}
                      style={{
                        marginTop: '0.75rem',
                        padding: '0.4rem 0.75rem',
                        fontSize: '0.75rem',
                        background: 'var(--accent-primary)',
                        color: 'white',
                        border: 'none',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontWeight: 600,
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => e.target.style.opacity = '0.85'}
                      onMouseLeave={(e) => e.target.style.opacity = '1'}
                    >
                      <i className="fas fa-eye" style={{ marginRight: '0.4rem' }} />
                      Show Examples
                    </button>
                    <div id="sentence-pattern-examples" style={{ display: 'none', marginTop: '0.75rem' }}>
                      <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-primary)' }}>
                        What Contributed to This Score:
                      </p>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                        <div style={{ padding: '0.5rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '6px', marginBottom: '0.5rem' }}>
                          <strong style={{ color: 'var(--text-primary)' }}>Sentence Length Variety:</strong>
                          <div style={{ marginTop: '0.25rem' }}>
                            Your sentences averaged <strong>{sentencePatternBreakdown.average_length}</strong> words with a standard deviation of <strong>{sentencePatternBreakdown.length_std}</strong>.
                          </div>
                          <div style={{ marginTop: '0.25rem' }}>
                            • <strong>{sentencePatternBreakdown.short_pct}%</strong> were short (≤8 words)
                          </div>
                          <div>
                            • <strong>{sentencePatternBreakdown.long_pct}%</strong> were long (≥25 words)
                          </div>
                        </div>
                        <div style={{ padding: '0.5rem', background: 'rgba(245, 158, 11, 0.1)', borderRadius: '6px' }}>
                          <strong style={{ color: 'var(--text-primary)' }}>💡 Tip:</strong>
                          <div style={{ marginTop: '0.25rem' }}>
                            Aim for 10-20 words per sentence on average with good variety. Too many short sentences feel choppy; too many long ones lose clarity.
                          </div>
                        </div>
                      </div>
                      {repetitionAlerts.length > 0 && (
                        <div style={{ marginTop: '0.75rem' }}>
                          <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-primary)' }}>
                            Repetition Patterns Detected:
                          </p>
                          <ul className="trend-list compact">
                            {repetitionAlerts.slice(0, 5).map((alert, idx) => {
                              if (typeof alert === 'string') {
                                return <li key={idx}>{alert}</li>
                              }
                              return (
                                <li key={idx} style={{ marginBottom: '0.4rem' }}>
                                  <strong style={{ color: 'var(--accent-warning)' }}>
                                    {alert.pattern ? `"${alert.pattern}"` : 'Pattern'}
                                  </strong>
                                  {alert.count && ` repeated ${alert.count} times`}
                                  {alert.example && (
                                    <div style={{ marginTop: '0.2rem', fontStyle: 'italic', fontSize: '0.7rem' }}>
                                      Example: "{alert.example}"
                                    </div>
                                  )}
                                </li>
                              )
                            })}
                          </ul>
                        </div>
                      )}
                      <button
                        onClick={() => {
                          document.getElementById('sentence-pattern-examples').style.display = 'none'
                        }}
                        style={{
                          marginTop: '0.75rem',
                          padding: '0.3rem 0.6rem',
                          fontSize: '0.7rem',
                          background: 'transparent',
                          color: 'var(--text-secondary)',
                          border: '1px solid var(--border-color)',
                          borderRadius: '4px',
                          cursor: 'pointer'
                        }}
                      >
                        Hide Examples
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="muted">Sentence pattern analysis unavailable.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function MetricLoading() {
  return (
    <div className="metric-loading">
      <span className="metric-spinner" />
      <span>Analyzing...</span>
    </div>
  )
}

export default AnalyticsTab

