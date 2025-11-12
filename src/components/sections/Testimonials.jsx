const Testimonials = () => {
  return (
    <section id="testimonials" className="testimonials">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Trusted by <span className="highlight-text">professionals</span> worldwide</h2>
          <p className="section-description">
            See what our users have to say about their experience.
          </p>
        </div>
        
        <div className="testimonials-grid">
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"It improved my presentation skills that I can easily present slides during IOE lectures with British accent (ps: join my coaching classes)"</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">
                <i className="fas fa-user"></i>
              </div>
              <div className="author-info">
                <h4>Dharamveer Saw</h4>
                <span>Student</span>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"The body language analysis helped me understand my non-verbal communication patterns. Game changer but still can't get any bitches"</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">
                <i className="fas fa-user"></i>
              </div>
              <div className="author-info">
                <h4>Ganesh Podeti</h4>
                <span>Developer</span>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"The AI-powered feedback helped me identify areas I never noticed. My confidence has improved significantly since using this platform."</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">
                <i className="fas fa-user"></i>
              </div>
              <div className="author-info">
                <h4>Sarah Chen</h4>
                <span>Business Professional</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Testimonials

