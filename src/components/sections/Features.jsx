const Features = () => {
  return (
    <section id="features" className="features">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">FEATURES THAT MAKE YOU <span className="highlight-text">UNSTOPPABLE</span></h2>
          <p className="section-description">
            Everything you need to perfect your presentations, powered by cutting-edge AI technology.
          </p>
        </div>
        
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-card-overlay"></div>
            <div className="feature-card-inner">
              <div className="feature-icon">
                <i className="fas fa-microphone"></i>
              </div>
              <h3>AI Speech Analysis</h3>
              <p>Advanced natural language processing analyzes your tone, pace, clarity, and filler words in real-time.</p>
              <div className="feature-highlight">
                <span>Real-time Processing</span>
              </div>
            </div>
          </div>
          
          <div className="feature-card">
            <div className="feature-card-overlay"></div>
            <div className="feature-card-inner">
              <div className="feature-icon">
                <i className="fas fa-eye"></i>
              </div>
              <h3>Body Language Tracking</h3>
              <p>Computer vision technology monitors your posture, gestures, and facial expressions for optimal non-verbal communication.</p>
              <div className="feature-highlight">
                <span>Advanced Computer Vision</span>
              </div>
            </div>
          </div>
          
          <div className="feature-card">
            <div className="feature-card-overlay"></div>
            <div className="feature-card-inner">
              <div className="feature-icon">
                <i className="fas fa-chart-line"></i>
              </div>
              <h3>Engagement Analytics</h3>
              <p>Track audience attention, reaction patterns, and overall presentation effectiveness with detailed metrics.</p>
              <div className="feature-highlight">
                <span>Detailed Analytics</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Features

