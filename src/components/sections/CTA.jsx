const CTA = () => {
  return (
    <section className="cta">
      <div className="cta-background">
        <img src="images/bg112.jpg" alt="Background" className="cta-bg-image" />
      </div>
      <div className="container">
        <div className="cta-content">
          <h2>Ready to transform your presentations?</h2>
          <p>Join thousands of professionals who have improved their presentation skills with Face2Phase.</p>
          <div className="cta-buttons">
            <button className="btn-primary btn-large btn-glassy">
              <i className="fas fa-rocket"></i>
              Start Free Trial
            </button>
            <button className="btn-secondary btn-large btn-glassy">
              <i className="fas fa-calendar"></i>
              Book a Demo
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}

export default CTA

