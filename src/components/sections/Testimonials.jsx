const Testimonials = () => {
  return (
    <section id="testimonials" className="testimonials">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">TRUSTED BY <span className="highlight-text">PROFESSIONALS</span> WORLDWIDE</h2>
          <p className="section-description">
            See what our users have to say about their experience.
          </p>
        </div>
        
        <div className="testimonials-grid">
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"Face2Phase transformed my presentation skills completely. The detailed feedback on my speaking pace and body language helped me deliver more confident presentations in my university lectures."</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">
                <i className="fas fa-user"></i>
              </div>
              <div className="author-info">
                <h4>Alex Thompson</h4>
                <span>Student</span>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"The body language analysis was a game-changer for me. I finally understand my non-verbal communication patterns and how they impact my presentations. Highly recommended for professionals looking to improve."</p>
            </div>
            <div className="testimonial-author">
              <div className="author-avatar">
                <i className="fas fa-user"></i>
              </div>
              <div className="author-info">
                <h4>Michael Rodriguez</h4>
                <span>Software Developer</span>
              </div>
            </div>
          </div>
          
          <div className="testimonial-card">
            <div className="testimonial-content">
              <p>"The AI-powered feedback helped me identify areas I never noticed before. My presentation confidence has improved significantly since using this platform. It's like having a personal coach available 24/7."</p>
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

