import { useState, useEffect, useRef, useMemo } from 'react'
import './TranscriptPanel.css'

const TranscriptPanel = ({ transcript, currentTime, onSeek, acousticFillers = [] }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredTranscript, setFilteredTranscript] = useState(transcript || [])
  const [userHasScrolled, setUserHasScrolled] = useState(false)
  const activeSegmentRef = useRef(null)
  const transcriptContentRef = useRef(null)
  const lastScrollTime = useRef(0)

  const getSegmentKey = (segment) => {
    if (!segment) return ''
    const startValue = typeof segment.start === 'number'
      ? segment.start
      : typeof segment.timestamp_seconds === 'number'
        ? segment.timestamp_seconds
        : null
    const endValue = typeof segment.end === 'number'
      ? segment.end
      : typeof segment.stop === 'number'
        ? segment.stop
        : null
    const timestampValue = typeof segment.timestamp === 'string' ? segment.timestamp : ''
    const textSnippet = typeof segment.text === 'string' ? segment.text.slice(0, 32) : ''
    return `${startValue ?? 'ns'}|${endValue ?? 'ne'}|${timestampValue}|${textSnippet}`
  }

  const segmentFillerCounts = useMemo(() => {
    if (!Array.isArray(transcript) || transcript.length === 0) {
      return new Map()
    }

    const map = new Map()

    // Filler words to detect in text
    const fillerWords = new Set([
      'um', 'uh', 'ah', 'er', 'hmm', 'erm', 'eh',
      'umm', 'uhh', 'ahh', 'ehh', 'err', 'uhm', 'ahm',
      'ummm', 'uhhh', 'ahhh', 'ehhh', 'errr', 'oh'
    ])

    // Count actual filler words in each segment's text
    transcript.forEach((segment) => {
      const text = segment.text || ''
      const words = text.toLowerCase().split(/\s+/)

      let count = 0
      words.forEach(word => {
        // Remove punctuation and check if it's a filler word
        const cleanWord = word.replace(/[.,!?;:]/g, '')
        if (fillerWords.has(cleanWord)) {
          count++
        }
      })

      if (count > 0) {
        const key = getSegmentKey(segment)
        map.set(key, count)
      }
    })

    return map
  }, [transcript])

  useEffect(() => {
    if (!transcript) return

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      const filtered = transcript.filter(segment =>
        segment.text.toLowerCase().includes(query)
      )
      setFilteredTranscript(filtered)
    } else {
      setFilteredTranscript(transcript)
    }
  }, [transcript, searchQuery])

  // Handle user scroll detection - AGGRESSIVE MODE
  useEffect(() => {
    const contentEl = transcriptContentRef.current
    if (!contentEl) return

    let scrollTimeout = null

    const handleScroll = (e) => {
      // Clear any pending timeout
      if (scrollTimeout) {
        clearTimeout(scrollTimeout)
      }

      // Mark as user-initiated scroll IMMEDIATELY
      const now = Date.now()
      if (now - lastScrollTime.current > 50) {
        setUserHasScrolled(true)
      }
    }

    const handleWheel = () => {
      // User is actively scrolling with mouse wheel
      setUserHasScrolled(true)
    }

    const handleTouchMove = () => {
      // User is actively scrolling with touch
      setUserHasScrolled(true)
    }

    contentEl.addEventListener('scroll', handleScroll, { passive: true })
    contentEl.addEventListener('wheel', handleWheel, { passive: true })
    contentEl.addEventListener('touchmove', handleTouchMove, { passive: true })

    return () => {
      contentEl.removeEventListener('scroll', handleScroll)
      contentEl.removeEventListener('wheel', handleWheel)
      contentEl.removeEventListener('touchmove', handleTouchMove)
      if (scrollTimeout) clearTimeout(scrollTimeout)
    }
  }, [])

  // Auto-scroll ONLY if user hasn't scrolled at all
  useEffect(() => {
    if (activeSegmentRef.current && !userHasScrolled) {
      lastScrollTime.current = Date.now()
      activeSegmentRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      })
    }
  }, [currentTime, userHasScrolled])

  const handleSegmentClick = (segment) => {
    if (onSeek && typeof segment.start === 'number') {
      // Reset user scroll flag when they click a segment
      setUserHasScrolled(false)
      onSeek(segment.start)
    }
  }

  const copyTranscript = () => {
    if (!transcript) return
    const fullText = transcript.map(seg => seg.text).join(' ')
    navigator.clipboard.writeText(fullText).then(() => {
      alert('Transcript copied to clipboard!')
    })
  }

  const isSegmentActive = (segment) => {
    const segStart = typeof segment.start === 'number' ? segment.start : segment.timestamp_seconds
    const segEnd = typeof segment.end === 'number' ? segment.end : segment.stop
    if (typeof segStart !== 'number' || typeof segEnd !== 'number') return false
    return currentTime >= segStart && currentTime <= segEnd
  }

  return (
    <div className="transcript-panel">
      <div className="transcript-header">
        <h3>Transcript</h3>
        <div className="transcript-actions">
          <input
            type="text"
            placeholder="Search transcript..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="transcript-search"
          />
          <button onClick={copyTranscript} className="copy-btn">
            <i className="fas fa-copy" /> Copy transcript
          </button>
        </div>
      </div>

      {userHasScrolled && (
        <div style={{
          padding: '0.5rem 1rem',
          background: 'rgba(59, 130, 246, 0.1)',
          borderBottom: '1px solid rgba(59, 130, 246, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)'
        }}>
          <span>
            <i className="fas fa-info-circle" style={{ marginRight: '0.5rem' }} />
            Auto-scroll disabled - you're browsing freely
          </span>
          <button
            onClick={() => setUserHasScrolled(false)}
            style={{
              padding: '0.25rem 0.5rem',
              fontSize: '0.7rem',
              background: 'var(--accent-primary)',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Re-enable auto-scroll
          </button>
        </div>
      )}

      <div className="transcript-content" ref={transcriptContentRef}>
        {filteredTranscript.length === 0 ? (
          <div className="transcript-empty">
            {searchQuery ? 'No results found' : 'No transcript available'}
          </div>
        ) : (
          filteredTranscript.map((segment, index) => {
            const isActive = isSegmentActive(segment)
            const segmentKey = getSegmentKey(segment)
            const fillerCount = segmentFillerCounts.get(segmentKey) || 0

            return (
              <div
                key={segmentKey || index}
                ref={isActive ? activeSegmentRef : null}
                className={`transcript-segment ${isActive ? 'active' : ''}`}
                onClick={() => handleSegmentClick(segment)}
              >
                <div className="transcript-segment-header">
                  <span className="transcript-timestamp">{segment.timestamp}</span>
                  {fillerCount > 0 && (
                    <span className="transcript-filler-chip">
                      {fillerCount} filler{fillerCount > 1 ? 's' : ''}
                    </span>
                  )}
                </div>
                <p className="transcript-text-content">{segment.text}</p>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

export default TranscriptPanel