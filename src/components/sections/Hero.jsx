const Hero = () => {
  return (
    <section className="hero">
      <div className="hero-background">
        <div className="blur-orbs">
          <div className="orb orb-1"></div>
          <div className="orb orb-2"></div>
          <div className="orb orb-3"></div>
          <div className="orb orb-4"></div>
          <div className="orb orb-5"></div>
        </div>
      </div>
      
      <div className="container">
        <div className="hero-content">
          <h1 className="hero-title">
            The truly <span className="highlight-text italic-text">limitless</span> presentation analysis.
          </h1>
          <p className="hero-description">
            Say goodbye to expensive presentation coaches, and hello to limitless, lightning-fast AI-powered feedback and analysis.
          </p>
          
          <div className="hero-buttons">
            <button className="btn-primary">
              <i className="fas fa-play"></i>
              Start Free Trial
            </button>
            <button className="btn-secondary">
              <i className="fas fa-calendar"></i>
              Book a Demo
            </button>
          </div>
          
          <div className="hero-stats">
            <div className="stat-item">
              <span className="stat-number">DEV</span>
              <span className="stat-label">Currently Developing</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">BETA</span>
              <span className="stat-label">Coming Soon</span>
            </div>
            <div className="stat-item">
              <span className="stat-number">2026</span>
              <span className="stat-label">Launch Year</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero

