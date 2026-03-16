import CardSwap, { Card } from '../CardSwap'

const HowItWorks = () => {
  return (
    <section id="how-it-works" className="how-it-works">
      <div className="container">
        <div className="section-header">
          <h2 className="section-title">HOW IT WORKS</h2>
          <p className="section-description">
            Get started in minutes with our streamlined 3-step process.
          </p>
        </div>
        
        <div className="how-it-works-layout">
          <div className="steps-info-text">
            <h3 className="steps-main-heading">From upload to insights in minutes</h3>
            <p className="steps-sub-heading">
              Face2Phase analyzes your delivery with executive-level benchmarks—no prep needed.
            </p>

            <div className="steps-details">
              <div className="step-detail">
                <h4 className="step-detail-title">Upload</h4>
                <p className="step-detail-text">
                  Upload your presentation video. We support all major formats including MP4, MOV, and AVI.
                  Maximum file size: 500MB.
                </p>
            </div>

              <div className="step-detail">
                <h4 className="step-detail-title">Analyze</h4>
                <p className="step-detail-text">
                  Our advanced AI processes your video, analyzing speech patterns, body language, and
                  engagement metrics in real-time.
                </p>
          </div>
          
              <div className="step-detail">
                <h4 className="step-detail-title">Insights</h4>
                <p className="step-detail-text">
                  Receive detailed feedback, actionable recommendations, and personalized insights to improve
                  your presentations.
                </p>
              </div>
            </div>
          </div>
          
          <div className="card-stack-container">
            <CardSwap
              cardDistance={50}
              verticalDistance={60}
              delay={4000}
              pauseOnHover={true}
              width={320}
              height={400}
              skewAmount={8}
            >
              <Card customClass="stack-card">
                <div className="card-header">
                  <span className="card-label">Insights</span>
                </div>
                <div className="card-gradient-bg">
                  <div className="large-number">3</div>
                </div>
              </Card>

              <Card customClass="stack-card">
                <div className="card-header">
                  <span className="card-label">Analyze</span>
                </div>
                <div className="card-gradient-bg">
                  <div className="large-number">2</div>
                </div>
              </Card>

              <Card customClass="stack-card featured">
                <div className="card-header">
                  <span className="card-label">Upload</span>
                </div>
                <div className="card-gradient-bg">
                  <div className="large-number">1</div>
            </div>
              </Card>
            </CardSwap>
          </div>
        </div>
      </div>
    </section>
  )
}

export default HowItWorks

