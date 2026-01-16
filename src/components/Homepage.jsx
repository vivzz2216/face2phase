import Navbar from './Navbar'
import Hero from './sections/Hero'
import Features from './sections/Features'
import HowItWorks from './sections/HowItWorks'
import Testimonials from './sections/Testimonials'
import CTA from './sections/CTA'
import Footer from './Footer'
import '../index.css' // Import global CSS for homepage styles

const Homepage = () => {
  return (
    <div className="homepage">
      <Navbar />
      <Hero />
      <Features />
      <HowItWorks />
      <Testimonials />
      <CTA />
      <Footer />
    </div>
  )
}

export default Homepage

