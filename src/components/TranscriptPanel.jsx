import { useState, useEffect, useRef, useMemo } from 'react'
import './TranscriptPanel.css'

const TranscriptPanel = ({ transcript, currentTime, onSeek, acousticFillers = [] }) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [filteredTranscript, setFilteredTranscript] = useState(transcript || [])
  const activeSegmentRef = useRef(null)

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

    const segments = transcript
    const map = new Map()

    const resolveSegmentStart = (seg) => {
      if (typeof seg.start === 'number') return seg.start
      if (typeof seg.timestamp_seconds === 'number') return seg.timestamp_seconds
      if (typeof seg.time === 'number') return seg.time
      return null
    }

    const resolveSegmentEnd = (index) => {
      const seg = segments[index]
      if (typeof seg.end === 'number') return seg.end
      if (typeof seg.stop === 'number') return seg.stop
      if (index + 1 < segments.length) {
        const next = segments[index + 1]
        const nextStart = resolveSegmentStart(next)
        if (typeof nextStart === 'number') return nextStart
      }
      const start = resolveSegmentStart(seg)
      return start !== null ? start + 0.5 : null
    }

    (acousticFillers || []).forEach((event) => {
      const eventStart = typeof event.start === 'number' ? event.start : null
      if (eventStart === null) return

      let matchedIndex = -1
      for (let i = 0; i < segments.length; i += 1) {
        const segStart = resolveSegmentStart(segments[i])
        const segEnd = resolveSegmentEnd(i)
        if (segStart === null) continue

        const effectiveEnd = segEnd !== null ? segEnd : segStart + 0.5
        const padding = 0.05
        if (eventStart >= segStart - padding && eventStart <= effectiveEnd + padding) {
          matchedIndex = i
          break
        }
      }

      if (matchedIndex >= 0) {
        const key = getSegmentKey(segments[matchedIndex])
        map.set(key, (map.get(key) || 0) + 1)
      }
    })

    return map
  }, [acousticFillers, transcript])

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

  useEffect(() => {
    if (activeSegmentRef.current) {
      activeSegmentRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      })
    }
  }, [currentTime])

  const handleSegmentClick = (segment) => {
    if (onSeek && typeof segment.start === 'number') {
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

      <div className="transcript-content">
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
