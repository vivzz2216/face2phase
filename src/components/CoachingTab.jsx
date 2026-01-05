import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { playPronunciation, stopPronunciation, isSpeechSynthesisAvailable } from '../lib/audioPlayer'
import './CoachingTab.css'

const CoachingTab = ({ feedback, reportData }) => {
  const [currentQuestionPage, setCurrentQuestionPage] = useState(1)
  const [pronunciationVisible, setPronunciationVisible] = useState(true)
  const [playingPronunciation, setPlayingPronunciation] = useState(null)
  
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      stopPronunciation()
    }
  }, [])
  
  const handlePlayPronunciation = async (word, correct) => {
    if (playingPronunciation === word) {
      stopPronunciation()
      setPlayingPronunciation(null)
      return
    }
    
    setPlayingPronunciation(word)
    try {
      // Use the actual word, not the formatted stress pattern
      // If correct contains formatting (like "tech-NOL-o-gy"), extract just the word
      let textToSpeak = word
      if (correct && correct !== word) {
        // Check if correct is a formatted stress pattern (contains hyphens or special chars)
        // If it looks like a stress pattern, use the original word instead
        if (correct.includes('-') || correct.includes('/') || correct.match(/[A-Z]{2,}/)) {
          textToSpeak = word // Use original word
        } else {
          textToSpeak = correct // Use correct if it's a simple word
        }
      }
      await playPronunciation(textToSpeak)
      setPlayingPronunciation(null)
    } catch (error) {
      console.error('Error playing pronunciation:', error)
      setPlayingPronunciation(null)
    }
  }

  if (!feedback && !reportData) {
    return <div className="coaching-empty">No feedback available</div>
  }

  // Get data from feedback first, then fallback to reportData
  const pronunciationData = feedback?.pronunciation || (reportData && (reportData.pronunciation || reportData.pronunciation_analysis)) || null
  const metrics = feedback?.metrics || {}
  const fillerMetrics = metrics.filler || {}
  const weakMetrics = metrics.weak_words || {}
  const concisenessMetrics = metrics.conciseness || {}
  const hasMetrics = Boolean(
    (fillerMetrics && (fillerMetrics.count !== undefined || fillerMetrics.percentage !== undefined)) ||
    (weakMetrics && (weakMetrics.count !== undefined || weakMetrics.percentage !== undefined)) ||
    (concisenessMetrics && (concisenessMetrics.score !== undefined || concisenessMetrics.notes))
  )

  const fallbackWordAnalysis = reportData?.word_analysis || {}
  const fallbackFillerCount = fallbackWordAnalysis?.filler_words?.filler_count || reportData?.filler_word_count || 0
  const fallbackTotalWords = fallbackWordAnalysis?.vocabulary?.total_words || reportData?.total_words || 0
  const fallbackWeakCount = fallbackWordAnalysis?.weak_words?.weak_word_count || 0
  const fillerPercentage = fillerMetrics.percentage !== undefined
    ? fillerMetrics.percentage
    : (fallbackTotalWords ? ((fallbackFillerCount / fallbackTotalWords) * 100).toFixed(1) : '0.0')
  const weakWordsPercentage = weakMetrics.percentage !== undefined
    ? weakMetrics.percentage
    : (fallbackTotalWords ? ((fallbackWeakCount / fallbackTotalWords) * 100).toFixed(1) : '0.0')

  const concisenessScore = concisenessMetrics.score !== undefined ? concisenessMetrics.score : null
  const concisenessNotes = concisenessMetrics.notes || null
  const summaryText = (() => {
    const summary = feedback?.summary
    if (typeof summary === 'string') {
      return summary
    }
    if (summary && typeof summary === 'object') {
      if (typeof summary.text === 'string') {
        return summary.text
      }
      if (Array.isArray(summary.points)) {
        return summary.points
          .filter((point) => typeof point === 'string')
          .join(' ')
      }
    }
    return ''
  })()
  const parsedFillerPercentage = typeof fillerPercentage === 'number'
    ? fillerPercentage
    : parseFloat(fillerPercentage)
  const summaryInsights = useMemo(() => {
    const insights = []
    const cleaned = summaryText.replace(/\s+/g, ' ').trim()
    if (cleaned) {
      const sentences = cleaned.split(/(?<=[.!?])\s+/).filter(Boolean)
      let combinedLength = 0
      for (const sentence of sentences) {
        if (combinedLength + sentence.length <= 240 || insights.length === 0) {
          insights.push(sentence)
          combinedLength += sentence.length
        } else {
          break
        }
        if (insights.length >= 2) break
      }
    }
    if (!insights.length && feedback?.strength?.message) {
      insights.push(feedback.strength.message)
    }
    if (feedback?.growth_areas?.length) {
      insights.push(`Focus next: ${feedback.growth_areas[0]}`)
    }
    if (!Number.isNaN(parsedFillerPercentage) && parsedFillerPercentage !== undefined) {
      insights.push(`Filler words: ${Number(parsedFillerPercentage).toFixed(1)}%`)
    }
    if (concisenessScore !== null && concisenessScore !== undefined) {
      insights.push(`Conciseness: ${Math.round(concisenessScore)}/100`)
    }
    return insights.slice(0, 3)
  }, [summaryText, feedback?.strength?.message, feedback?.growth_areas, parsedFillerPercentage, concisenessScore])
  
  // Debug: Log available data
  if (process.env.NODE_ENV === 'development') {
    console.log('CoachingTab - Feedback:', feedback)
    console.log('CoachingTab - ReportData:', reportData)
    console.log('CoachingTab - Pronunciation:', pronunciationData)
  }

  const questions = feedback?.follow_up_questions || []
  const questionsPerPage = 1
  const totalPages = Math.ceil(questions.length / questionsPerPage)
  const currentQuestions = questions.slice(
    (currentQuestionPage - 1) * questionsPerPage,
    currentQuestionPage * questionsPerPage
  )
  const structureMetrics = reportData?.text_analysis?.structure_metrics || {}
  const advancedTextMetrics = reportData?.text_analysis?.advanced_text_metrics || reportData?.text_analytics || {}
  const fillerAnalysis = feedback?.metrics?.filler || reportData?.filler_analysis || {}
  const pronunciationIssues = pronunciationData?.issues || []
  const visualMetrics = reportData?.visual_analytics || {}
  const fillerPercentagePrecise = typeof fillerAnalysis.percentage === 'number'
    ? fillerAnalysis.percentage
    : (typeof reportData?.filler_word_ratio === 'number' ? reportData.filler_word_ratio * 100 : null)

  const strengthTable = useMemo(() => {
    const rows = []
    const topKeywords = Array.isArray(advancedTextMetrics.keyword_details)
      ? advancedTextMetrics.keyword_details.slice(0, 3)
      : []
    if (topKeywords.length) {
      rows.push({
        aspect: 'Topic Knowledge',
        observation: `Highlighted ${topKeywords.length >= 3 ? 'multiple' : 'several'} focus areas such as ${topKeywords.map((item) => item.word).join(', ')}.`,
        feedback: 'Shows command of core themes—keep weaving these references into your narrative.'
      })
    }
    if (feedback?.strength?.message) {
      rows.push({
        aspect: 'Enthusiasm / Intent',
        observation: feedback.strength.message,
        feedback: 'Motivated delivery engages listeners—retain this positive intent.'
      })
    }
    if (structureMetrics?.total_sentences) {
      rows.push({
        aspect: 'Sequential Flow',
        observation: `Spoke in ${structureMetrics.total_sentences} sentences with step-by-step progression.`,
        feedback: 'Logical ordering builds clarity; keep introducing topics before diving deep.'
      })
    }
    if (Array.isArray(pronunciationIssues) && pronunciationIssues.length === 0 && pronunciationData?.summary) {
      rows.push({
        aspect: 'Attempted Explanation',
        observation: pronunciationData.summary,
        feedback: 'Breaking terms down adds depth—great technique for educating your audience.'
      })
    } else if (!pronunciationData?.summary && rows.length < 3) {
      rows.push({
        aspect: 'Narrative Focus',
        observation: 'Introduced the topic, provided context, then expanded with supporting details.',
        feedback: 'Clear progression keeps audiences oriented. Continue anchoring explanations in context.'
      })
    }
    return rows
  }, [advancedTextMetrics.keyword_details, feedback?.strength?.message, pronunciationData, pronunciationIssues, structureMetrics?.total_sentences])

  const weaknessTable = useMemo(() => {
    const rows = []
    if (fillerPercentagePrecise !== null) {
      rows.push({
        category: 'Filler Words / Disfluency',
        issue: `Filler usage at ~${fillerPercentagePrecise.toFixed(1)}% of spoken words.`,
        advice: 'Practice deliberate pauses. Aim to keep fillers below 5% by rehearsing with recorded playback.'
      })
    }
    if (structureMetrics?.sentence_variety !== undefined) {
      rows.push({
        category: 'Sentence Structure',
        issue: `Sentence variety score at ${structureMetrics.sentence_variety?.toFixed?.(1) ?? structureMetrics.sentence_variety}.`,
        advice: 'Outline sentences with varied openings and lengths to maintain rhythm and clarity.'
      })
    }
    if (typeof advancedTextMetrics?.compression_ratio === 'number') {
      rows.push({
        category: 'Clarity / Brevity',
        issue: `Compression ratio ${advancedTextMetrics.compression_ratio} suggests repeated phrasing.`,
        advice: 'Replace repeated descriptors with fresh wording (e.g., “ancient classical form” instead of repeating “very old and traditional”).'
      })
    }
    if (typeof reportData?.voice_confidence === 'number') {
      rows.push({
        category: 'Voice Flow / Confidence',
        issue: `Voice confidence scored ${Math.round(reportData.voice_confidence)}/100 with noticeable hesitations.`,
        advice: 'Rehearse in short sections focusing on smooth transitions and steady breathing.'
      })
    }
    if (Array.isArray(pronunciationIssues) && pronunciationIssues.length > 0) {
      rows.push({
        category: 'Pronunciation',
        issue: `Mispronounced terms such as ${(pronunciationIssues[0].word || '').toString()}.`,
        advice: 'Practice slowly: break the word into syllables (e.g., Bha–ra–ta–na–tyam) and repeat aloud.'
      })
    }
    if (!reportData?.closing_confidence && reportData?.summary) {
      rows.push({
        category: 'Closure / Ending',
        issue: 'Closing felt abrupt with filler words.',
        advice: 'Plan a clean sign-off (e.g., “Thank you for listening” or “That was a brief introduction to…”).'
      })
    }
    if (visualMetrics?.tension_summary?.avg_eye_contact_pct !== undefined) {
      rows.push({
        category: 'Engagement / Variation',
        issue: 'Delivery tone could benefit from additional variation.',
        advice: 'Add emphasis on keywords and modulate pitch slightly to keep attention on cultural highlights.'
      })
    }
    return rows
  }, [advancedTextMetrics?.compression_ratio, fillerPercentagePrecise, pronunciationIssues, reportData?.summary, reportData?.voice_confidence, structureMetrics?.sentence_variety, visualMetrics?.tension_summary?.avg_eye_contact_pct])

  const strengthSummary = useMemo(() => {
    const points = []
    if (strengthTable.length) {
      points.push('Subject knowledge and sequencing already create a solid foundation.')
    }
    if (advancedTextMetrics?.keyword_details?.length) {
      points.push('Mentioning specific cultural references adds authenticity.')
    }
    if (feedback?.strength?.message) {
      points.push('Audience can feel your intent to inform and educate.')
    }
    return points.slice(0, 3)
  }, [strengthTable, advancedTextMetrics?.keyword_details?.length, feedback?.strength?.message])

  const weaknessSummary = useMemo(() => {
    const points = []
    if (fillerPercentagePrecise !== null && fillerPercentagePrecise > 5) {
      points.push('Reduce filler words to sound more confident and polished.')
    }
    if (pronunciationIssues.length) {
      points.push('Tidy pronunciation of key terms to reinforce credibility.')
    }
    if (typeof reportData?.voice_confidence === 'number' && reportData.voice_confidence < 75) {
      points.push('Boost vocal flow with paced breathing and phrase linking.')
    }
    return points.slice(0, 4)
  }, [fillerPercentagePrecise, pronunciationIssues.length, reportData?.voice_confidence])

  return (
    <div className="coaching-tab">
      {/* Strength */}
      {feedback.strength && (
        <motion.div
          className="coaching-section strength-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="section-header strength-header">
            <i className="fas fa-lightbulb" />
            <span className="section-label">Strength</span>
          </div>
          <div className="section-content">
          {strengthTable.length > 0 && (
            <div className="coaching-table-block">
              <h3 className="coaching-table-title">💪 Strengths</h3>
              <div className="coaching-table-wrapper">
                <table className="coaching-table">
                  <thead>
                    <tr>
                      <th>Aspect</th>
                      <th>Observation</th>
                      <th>Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strengthTable.map((row, idx) => (
                      <tr key={`strength-${row.aspect}-${idx}`}>
                        <td>{row.aspect}</td>
                        <td>{row.observation}</td>
                        <td>{row.feedback}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {strengthSummary.length > 0 && (
                <div className="coaching-summary">
                  <p className="coaching-summary-title">🧠 Summary</p>
                  <ul className="coaching-summary-list">
                    {strengthSummary.map((item, idx) => (
                      <li key={`strength-summary-${idx}`}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          {!strengthTable.length && feedback.strength?.message && <p>{feedback.strength.message}</p>}
          </div>
        </motion.div>
      )}

      {/* Growth Area */}
      {feedback.growth_areas && feedback.growth_areas.length > 0 && (
        <motion.div
          className="coaching-section growth-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="section-header growth-header">
            <i className="fas fa-chart-line" />
            <span className="section-label">Growth Area</span>
          </div>
          <div className="section-content">
            {weaknessTable.length > 0 && (
              <div className="coaching-table-block">
                <h3 className="coaching-table-title warning">⚠️ Weaknesses & Precise Improvement Advice</h3>
                <div className="coaching-table-wrapper">
                  <table className="coaching-table">
                    <thead>
                      <tr>
                        <th>Category</th>
                        <th>Detected Issue</th>
                        <th>Precise Advice</th>
                      </tr>
                    </thead>
                    <tbody>
                      {weaknessTable.map((row, idx) => (
                        <tr key={`weakness-${row.category}-${idx}`}>
                          <td>{row.category}</td>
                          <td>{row.issue}</td>
                          <td>{row.advice}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {weaknessSummary.length > 0 && (
                  <div className="coaching-summary">
                    <p className="coaching-summary-title warning">🧠 Summary</p>
                    <ul className="coaching-summary-list">
                      {weaknessSummary.map((item, idx) => (
                        <li key={`weakness-summary-${idx}`}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}

            {(!weaknessTable.length && feedback.growth_areas.length > 0) && (
            <ul className="growth-list">
              {feedback.growth_areas.map((area, idx) => (
                <li key={idx}>{area}</li>
              ))}
            </ul>
            )}
          </div>
        </motion.div>
      )}

      {/* Follow-up Questions */}
      {questions.length > 0 && (
        <motion.div
          className="coaching-section questions-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="section-header questions-header">
            <i className="fas fa-question-circle" />
            <span className="section-label">Follow-up Questions</span>
          </div>
          <div className="section-content">
            <p className="questions-intro">Here are some follow-up questions you should be prepared for:</p>
            {currentQuestions.map((q, idx) => (
              <div key={idx} className="question-item">
                <div className="question-text">{q.question}</div>
                {q.timestamp && (
                  <div className="question-timestamp">{q.timestamp}</div>
                )}
              </div>
            ))}
            {totalPages > 1 && (
              <div className="questions-pagination">
                <button
                  className="pagination-btn"
                  onClick={() => setCurrentQuestionPage(prev => Math.max(1, prev - 1))}
                  disabled={currentQuestionPage === 1}
                >
                  ←
                </button>
                <span className="pagination-info">{currentQuestionPage}/{totalPages}</span>
                <button
                  className="pagination-btn"
                  onClick={() => setCurrentQuestionPage(prev => Math.min(totalPages, prev + 1))}
                  disabled={currentQuestionPage === totalPages}
                >
                  →
                </button>
              </div>
            )}
            <button className="practice-btn">
              <i className="fas fa-play" /> Practise
            </button>
          </div>
        </motion.div>
      )}

      {/* Metrics */}
      {hasMetrics && (
        <motion.div
          className="coaching-section metrics-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <div className="section-header metrics-header">
            <i className="fas fa-chart-pie" />
            <span className="section-label">Metrics</span>
          </div>
          <div className="section-content">
            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Filler Words</span>
                <span className="metric-value">
                  {fillerPercentage !== undefined ? `${fillerPercentage}%` : '—'}
                </span>
                {fillerMetrics.count !== undefined && (
                  <span className="metric-detail">{fillerMetrics.count} detected</span>
                )}
              </div>
              <div className="metric-item">
                <span className="metric-label">Weak Words</span>
                <span className="metric-value">
                  {weakWordsPercentage !== undefined ? `${weakWordsPercentage}%` : '—'}
                </span>
                {weakMetrics.count !== undefined && (
                  <span className="metric-detail">{weakMetrics.count} flagged</span>
                )}
              </div>
              <div className="metric-item">
                <span className="metric-label">Conciseness</span>
                <span className="metric-value">
                  {concisenessScore !== null ? `${Math.round(concisenessScore)} / 100` : '—'}
                </span>
                {concisenessNotes && (
                  <span className="metric-detail">{concisenessNotes}</span>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Summary */}
      {summaryInsights.length > 0 && (
        <motion.div
          className="coaching-section summary-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <div className="section-header summary-header">
            <i className="fas fa-file-alt" />
            <span className="section-label">Summary</span>
          </div>
          <div className="section-content">
            <ul className="summary-list">
              {summaryInsights.map((insight, idx) => (
                <li key={idx} className="summary-text">{insight}</li>
              ))}
            </ul>
          </div>
        </motion.div>
      )}

      {/* Pronunciation */}
      {pronunciationData && (
        <motion.div
          className="coaching-section pronunciation-section"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
        >
          <div className="section-header pronunciation-header">
            <div className="header-left">
              <i className="fas fa-volume-up" />
              <span className="section-label">Pronunciation</span>
            </div>
            <button
              className="hide-btn"
              onClick={() => setPronunciationVisible(!pronunciationVisible)}
            >
              {pronunciationVisible ? 'Hide' : 'Show'}
            </button>
          </div>
          {pronunciationVisible && (
            <div className="section-content">
              {pronunciationData.summary && (
                <p className="pronunciation-advice">{pronunciationData.summary}</p>
              )}
              {pronunciationData.issues && pronunciationData.issues.length > 0 ? (
                <div className="pronunciation-issues">
                  {pronunciationData.issues.map((issue, idx) => (
                    <div key={idx} className="pronunciation-item">
                      {issue.tip && (
                        <p className="pronunciation-context">{issue.tip}</p>
                      )}
                      <div className="pronunciation-comparison">
                        <span className="pronunciation-incorrect">{issue.word}</span>
                        {issue.correct && (
                          <>
                            <span className="pronunciation-arrow">→</span>
                            <span className="pronunciation-correct">{issue.correct}</span>
                          </>
                        )}
                      </div>
                      {isSpeechSynthesisAvailable() && (
                        <button
                          className="play-pronunciation-btn"
                          onClick={() => handlePlayPronunciation(issue.word, issue.correct)}
                          style={{ marginTop: '0.5rem' }}
                        >
                          <i className={`fas fa-${playingPronunciation === issue.word ? 'stop' : 'play'}`} />
                          {playingPronunciation === issue.word ? ' Stop' : ' Play'}
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
                  No pronunciation issues detected. Great job!
                </p>
              )}
            </div>
          )}
        </motion.div>
      )}
    </div>
  )
}

export default CoachingTab

