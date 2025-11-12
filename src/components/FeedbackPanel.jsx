import { useState } from 'react'
import CoachingTab from './CoachingTab'
import AnalyticsTab from './AnalyticsTab'
import './FeedbackPanel.css'

const FeedbackPanel = ({ feedback, reportData }) => {
  const [activeTab, setActiveTab] = useState('coaching')
  
  // Calculate actual coaching count from feedback sections
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

  const copyFeedback = () => {
    // Create text version of feedback
    let feedbackText = '=== PRESENTATION FEEDBACK ===\n\n'
    
    if (feedback?.strength) {
      feedbackText += `STRENGTH:\n${feedback.strength.message}\n\n`
    }
    
    if (feedback?.growth_areas?.length > 0) {
      feedbackText += `GROWTH AREAS:\n`
      feedback.growth_areas.forEach((area, idx) => {
        feedbackText += `${idx + 1}. ${area}\n`
      })
      feedbackText += '\n'
    }
    
    if (feedback?.follow_up_questions?.length > 0) {
      feedbackText += `FOLLOW-UP QUESTIONS:\n`
      feedback.follow_up_questions.forEach((q, idx) => {
        feedbackText += `${idx + 1}. ${q.question} (${q.timestamp})\n`
      })
      feedbackText += '\n'
    }
    
    if (feedback?.summary?.points?.length > 0) {
      feedbackText += `SUMMARY:\n`
      feedback.summary.points.forEach((point, idx) => {
        feedbackText += `• ${point}\n`
      })
    }
    
    navigator.clipboard.writeText(feedbackText).then(() => {
      alert('Feedback copied to clipboard!')
    })
  }

  return (
    <div className="feedback-panel">
      <div className="feedback-tabs">
        <button
          className={`tab-btn ${activeTab === 'coaching' ? 'active' : ''}`}
          onClick={() => setActiveTab('coaching')}
        >
          <i className="fas fa-comments" /> Coaching ({coachingCount})
        </button>
        <button
          className={`tab-btn ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <i className="fas fa-chart-bar" /> Analytics
        </button>
        <button className="copy-feedback-btn" onClick={copyFeedback}>
          <i className="fas fa-copy" /> Copy Feedback
        </button>
      </div>

      <div className="feedback-content">
        {activeTab === 'coaching' ? (
          <CoachingTab feedback={feedback} />
        ) : (
          <AnalyticsTab reportData={reportData} />
        )}
      </div>
    </div>
  )
}

export default FeedbackPanel

