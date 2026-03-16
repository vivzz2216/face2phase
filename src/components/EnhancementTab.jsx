import { useState, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import './EnhancementTab.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const EnhancementTab = ({ sessionId, reportData }) => {
  const [transcriptView, setTranscriptView] = useState('original') // 'original' or 'enhanced'
  const [vocabularyExpanded, setVocabularyExpanded] = useState({})
  const [loading, setLoading] = useState(false)
  const [transcriptData, setTranscriptData] = useState(null)
  const [vocabularyData, setVocabularyData] = useState(null)
  const [transcript, setTranscript] = useState('')

  useEffect(() => {
    loadEnhancementData()
  }, [sessionId, reportData])

  const loadEnhancementData = async () => {
    try {
      // Get transcript first
      const transcriptResponse = await fetch(`${API_BASE_URL}/api/video/${sessionId}/transcript`)
      if (transcriptResponse.ok) {
        const transcriptResult = await transcriptResponse.json()
        const transcriptText = transcriptResult.transcript?.map(t => t.text).join(' ') || ''
        setTranscript(transcriptText)
        
        // Get existing enhancement data from reportData if available
        if (reportData) {
          if (reportData.transcript_improvement) {
            setTranscriptData(reportData.transcript_improvement)
          }
          if (reportData.vocabulary_enhancements) {
            setVocabularyData(reportData.vocabulary_enhancements)
          }
        }
        
        // If no data exists, generate it
        if (!transcriptData || !vocabularyData) {
          await generateEnhancements(transcriptText)
        }
      }
    } catch (error) {
      console.error('Error loading enhancement data:', error)
    }
  }

  const generateEnhancements = async (transcriptText) => {
    if (!transcriptText || !transcriptText.trim()) {
      return
    }

    setLoading(true)
    try {
      // Generate transcript enhancement
      const transcriptResponse = await fetch(`${API_BASE_URL}/api/video/${sessionId}/enhance-transcript`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript: transcriptText })
      })

      if (transcriptResponse.ok) {
        const transcriptResult = await transcriptResponse.json()
        setTranscriptData(transcriptResult)
      }

      // Generate vocabulary enhancement
      const vocabResponse = await fetch(`${API_BASE_URL}/api/video/${sessionId}/enhance-vocabulary`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ transcript: transcriptText })
      })

      if (vocabResponse.ok) {
        const vocabResult = await vocabResponse.json()
        setVocabularyData(vocabResult)
      }
    } catch (error) {
      console.error('Error generating enhancements:', error)
    } finally {
      setLoading(false)
    }
  }

  // Use reportData if available, otherwise use state
  const transcriptImprovement = transcriptData || reportData?.transcript_improvement || null
  const vocabularyEnhancements = vocabularyData || reportData?.vocabulary_enhancements || null
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
        if (seen.has(key)) continue
        seen.add(key)
        enhancements.push(suggestion)
      }
    })

    return enhancements.slice(0, 5)
  }, [transcript])

  const displayVocabularyEnhancements = useMemo(() => {
    if (vocabularyEnhancements?.enhancements?.length) {
      return vocabularyEnhancements.enhancements
    }
    return autoVocabularyEnhancements
  }, [autoVocabularyEnhancements, vocabularyEnhancements])

  const totalVocabularySuggestions = vocabularyEnhancements?.total_suggestions ?? displayVocabularyEnhancements.length

  if (loading) {
    return (
      <div className="enhancement-loading">
        <div className="loading-spinner">
          <i className="fas fa-spinner fa-spin" />
        </div>
        <p>Generating improvements...</p>
      </div>
    )
  }

  return (
    <div className="enhancement-tab">
      {/* Transcript Improvement Section */}
      <motion.div
        className="enhancement-section transcript-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="section-header">
          <i className="fas fa-magic" />
          <span className="section-label">How You Could Have Spoken Better</span>
        </div>
        <div className="section-content">
          {transcriptImprovement && transcriptImprovement.success !== false ? (
            <>
              <div className="transcript-toggle">
                <button
                  className={`toggle-btn ${transcriptView === 'original' ? 'active' : ''}`}
                  onClick={() => setTranscriptView('original')}
                >
                  <i className="fas fa-file-alt" /> Your Original
                </button>
                <button
                  className={`toggle-btn ${transcriptView === 'enhanced' ? 'active' : ''}`}
                  onClick={() => setTranscriptView('enhanced')}
                >
                  <i className="fas fa-star" /> AI Enhanced
                </button>
              </div>
              
              <div className="transcript-display">
                {transcriptView === 'original' 
                  ? (transcriptImprovement.original || transcript || 'No transcript available')
                  : (transcriptImprovement.enhanced || transcriptImprovement.original || 'No enhanced version available')
                }
              </div>
              
              {transcriptImprovement.key_changes && transcriptImprovement.key_changes.length > 0 && (
                <div className="improvements-list">
                  <p className="improvements-title">Key Improvements:</p>
                  <ul>
                    {transcriptImprovement.key_changes.map((change, idx) => (
                      <li key={idx}>
                        <i className="fas fa-check-circle" />
                        {change}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <div className="no-data">
              <i className="fas fa-info-circle" />
              <p>Transcript enhancement will be available after analysis is complete.</p>
            </div>
          )}
        </div>
      </motion.div>

      {/* Vocabulary Enhancement Section */}
      <motion.div
        className="enhancement-section vocabulary-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="section-header">
          <i className="fas fa-book-reader" />
          <span className="section-label">Vocabulary Enhancement</span>
        </div>
        <div className="section-content">
          {displayVocabularyEnhancements.length > 0 ? (
            <>
              <p className="enhancement-intro">
                Replace informal or weak words with more professional alternatives:
              </p>
              
              <div className="vocabulary-enhancements">
                {displayVocabularyEnhancements.map((enhancement, idx) => {
                  const isExpanded = vocabularyExpanded[idx] || false
                  return (
                    <div key={idx} className="vocabulary-card">
                      <div 
                        className="vocabulary-card-header"
                        onClick={() => setVocabularyExpanded({ ...vocabularyExpanded, [idx]: !isExpanded })}
                      >
                        <div className="word-comparison">
                          <span className="word-original">"{enhancement.word}"</span>
                          <i className="fas fa-arrow-right" />
                          <span className="word-suggestion">{enhancement.suggestions?.[0] || 'N/A'}</span>
                        </div>
                        <i className={`fas fa-chevron-${isExpanded ? 'up' : 'down'}`} />
                      </div>
                      
                      {isExpanded && (
                        <div className="vocabulary-card-content">
                          {enhancement.context && (
                            <div className="context-example">
                              <span className="context-label">Context:</span>
                              <span className="context-text">"{enhancement.context}"</span>
                            </div>
                          )}
                          
                          {enhancement.suggestions && enhancement.suggestions.length > 1 && (
                            <div className="other-options">
                              <span className="options-label">Other options:</span>
                              <div className="options-list">
                                {enhancement.suggestions.slice(1).map((suggestion, sIdx) => (
                                  <span key={sIdx} className="option-tag">
                                    {suggestion}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {enhancement.reason && (
                            <div className="enhancement-reason">
                              <i className="fas fa-lightbulb" />
                              {enhancement.reason}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </>
          ) : (
            <div className="no-data">
              <i className="fas fa-check-circle" />
              <p>Great job! Your vocabulary is already professional and formal.</p>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}

export default EnhancementTab








