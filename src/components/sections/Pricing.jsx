const Pricing = () => {
  return (
    <section id="pricing" className="pricing">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">Pricing that's so <span className="highlight-text">simple</span>.</h2>
          <p className="section-description">
            Choose the plan that fits your needs. No hidden fees, no surprises.
          </p>
        </div>
        
        <div className="pricing-grid">
          <div className="pricing-card">
            <div className="pricing-card-overlay"></div>
            <div className="pricing-card-inner">
              <div className="pricing-header">
                <h3>Free</h3>
                <div className="price">
                  <span className="currency">₹</span>
                  <span className="amount">0</span>
                  <span className="period">/month</span>
                </div>
                <p className="price-description">Perfect for getting started</p>
              </div>
              
              <div className="pricing-features">
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>3 video analyses per month</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Basic AI analysis</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Speech pattern insights</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Email support</span>
                </div>
              </div>
              
              <div className="pricing-button-container">
                <button className="btn-secondary btn-full btn-glassy">Get Started</button>
              </div>
            </div>
          </div>
          
          <div className="pricing-card featured">
            <div className="pricing-card-overlay"></div>
            <div className="pricing-card-inner">
              <div className="popular-badge">
                <i className="fas fa-star"></i>
                Most Popular
              </div>
              
              <div className="pricing-header">
                <h3>Professional</h3>
                <div className="price">
                  <span className="currency">₹</span>
                  <span className="amount">8,250</span>
                  <span className="period">/month</span>
                </div>
                <p className="price-description">Perfect for professionals and small teams</p>
              </div>
              
              <div className="pricing-features">
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Unlimited video analyses</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Advanced AI analysis</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Body language tracking</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Engagement metrics</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Progress tracking</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Priority support</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Export reports</span>
                </div>
              </div>
              
              <div className="pricing-button-container">
                <button className="btn-primary btn-full btn-glassy">Get Started</button>
              </div>
              <p className="pricing-note">
                <i className="fas fa-shield-alt"></i>
                Cancel anytime
              </p>
            </div>
          </div>
          
          <div className="pricing-card">
            <div className="pricing-card-overlay"></div>
            <div className="pricing-card-inner">
              <div className="pricing-header">
                <h3>Business</h3>
                <div className="price">
                  <span className="currency">₹</span>
                  <span className="amount">24,900</span>
                  <span className="period">/month</span>
                </div>
                <p className="price-description">Perfect for large teams and enterprises</p>
              </div>
              
              <div className="pricing-features">
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Everything in Professional</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Team collaboration</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Custom integrations</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Advanced analytics</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>API access</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>Dedicated support</span>
                </div>
                <div className="feature-item">
                  <i className="fas fa-check"></i>
                  <span>SLA guarantee</span>
                </div>
              </div>
              
              <div className="pricing-button-container">
                <button className="btn-secondary btn-full btn-glassy">Contact Sales</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Pricing

