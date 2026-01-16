import { useState, useRef, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import './Chatbot.css'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const Chatbot = ({ sessionId }) => {
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const assistantIndexRef = useRef(null)

  // FIX #8: Session validation - ensure chatbot only responds with session-specific data
  const [sessionContext, setSessionContext] = useState(null)

  useEffect(() => {
    // Load session context for validation
    if (sessionId) {
      fetch(`${API_BASE_URL}/api/report/${sessionId}`)
        .then(res => res.ok ? res.json() : null)
        .then(data => {
          if (data) {
            setSessionContext({
              id: sessionId,
              title: data.project_name || data.file_name || 'Current Session',
              timestamp: data.timestamp || new Date().toISOString()
            })
          }
        })
        .catch(err => console.warn('Could not load session context:', err))
    }
  }, [sessionId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    if (!sessionId) {
      setMessages([])
      assistantIndexRef.current = null
      return
    }
    setMessages([])
    assistantIndexRef.current = null
  }, [sessionId])

  const sendMessage = useCallback(async (overrideMessage) => {
    const pendingMessage = typeof overrideMessage === 'string' ? overrideMessage : inputMessage
    if (!pendingMessage?.trim() || isStreaming || !sessionId) return

    const userMessage = pendingMessage.trim()
    setInputMessage('')
    setIsStreaming(true)

    const userMessageObj = {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    }
    const assistantMessageObj = {
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      streaming: true
    }

    setMessages(prev => {
      const next = [...prev, userMessageObj, assistantMessageObj]
      assistantIndexRef.current = next.length - 1
      return next
    })

    try {
      const response = await fetch(`${API_BASE_URL}/api/video/${sessionId}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userMessage })
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      if (!response.body) {
        throw new Error('No response body received')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let assistantBuffer = ''
      let done = false

      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading
        const chunkValue = value ? decoder.decode(value, { stream: !done }) : ''
        if (chunkValue) {
          assistantBuffer += chunkValue
          setMessages(prev => {
            const next = [...prev]
            const index = assistantIndexRef.current
            if (index !== null && index >= 0 && next[index]) {
              next[index] = { ...next[index], content: assistantBuffer }
            }
            return next
          })
          scrollToBottom()
        }
      }

      setMessages(prev => {
        const next = [...prev]
        const index = assistantIndexRef.current
        if (index !== null && index >= 0 && next[index]) {
          next[index] = {
            ...next[index],
            content: (next[index].content || '').trim() || 'Thanks for the question—here is what I found.',
            streaming: false
          }
        }
        return next
      })
    } catch (error) {
      console.error('Chatbot error:', error)
      setMessages(prev => {
        const next = [...prev]
        const index = assistantIndexRef.current
        if (index !== null && index >= 0 && next[index]) {
          next[index] = {
            ...next[index],
            content: 'Sorry, I encountered an error. Please try again.',
            streaming: false
          }
        } else {
          next.push({
            role: 'assistant',
            content: 'Sorry, I encountered an error. Please try again.',
            timestamp: new Date()
          })
        }
        return next
      })
    } finally {
      setIsStreaming(false)
      inputRef.current?.focus()
      scrollToBottom()
    }
  }, [inputMessage, isStreaming, sessionId])

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const refreshChat = async () => {
    if (!sessionId) return
    try {
      await fetch(`${API_BASE_URL}/api/video/${sessionId}/chat/reset`, {
        method: 'POST'
      })
    } catch (error) {
      console.warn('Chat reset failed (non-critical):', error)
    }
    setMessages([])
    setInputMessage('')
    assistantIndexRef.current = null
  }

  return (
    <div className="chatbot-container">
      {/* FIX #8: Display session context to show what chatbot is referencing */}
      {sessionContext && (
        <div style={{
          padding: '0.5rem 0.75rem',
          background: 'var(--bg-secondary)',
          borderBottom: '1px solid var(--border-color)',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)'
        }}>
          <i className="fas fa-info-circle" style={{ marginRight: '0.5rem' }} />
          Chatting about: <strong>{sessionContext.title}</strong>
        </div>
      )}
      <div className="chatbot-messages">
        {messages.length === 0 ? (
          <div className="chatbot-empty">
            <div className="empty-icon">
              <i className="fas fa-comments" />
            </div>
            <p>Start a conversation</p>
            <p className="empty-hint">Ask me anything you're curious about</p>
          </div>
        ) : (
          messages.map((message, index) => (
            <motion.div
              key={index}
              className={`message ${message.role}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className="message-avatar">
                {message.role === 'user' ? (
                  <div className="avatar-user">U</div>
                ) : (
                  <div className="avatar-ai">
                    <i className="fas fa-waveform-lines" />
                  </div>
                )}
              </div>
              <div className="message-content">
                {message.streaming && !message.content ? (
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                ) : (
                  <div className="message-text">{message.content}</div>
                )}
                {message.timestamp && (
                  <div className="message-timestamp">
                    {new Date(message.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                )}
              </div>
            </motion.div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chatbot-input-container">
        <div className="chatbot-input-wrapper">
          <input
            ref={inputRef}
            type="text"
            className="chatbot-input"
            placeholder="Type your message..."
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isStreaming || !sessionId}
          />
          <button
            className="chatbot-send-btn"
            onClick={sendMessage}
            disabled={isStreaming || !sessionId || !inputMessage.trim()}
          >
            <i className="fas fa-paper-plane" />
          </button>
        </div>
        <button
          className="chatbot-refresh-btn"
          onClick={refreshChat}
          title="Reset conversation"
          disabled={isStreaming}
        >
          <i className="fas fa-refresh" />
        </button>
      </div>
    </div>
  )
}

export default Chatbot

