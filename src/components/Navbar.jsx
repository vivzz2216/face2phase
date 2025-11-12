import { useState, useEffect, useRef } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false)
  const [file, setFile] = useState(null)
  const fileInputRef = useRef(null)
  const navigate = useNavigate()
  const location = useLocation()
  const { user } = useAuth()
  
  // Only show upload button on dashboard page
  const isDashboardPage = location.pathname === '/dashboard' || location.pathname.startsWith('/dashboard')

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50)
    }

    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      // Navigate to dashboard - user can upload there
      navigate('/dashboard')
      // Store file info in sessionStorage as fallback
      try {
        sessionStorage.setItem('fileToUpload', JSON.stringify({
          name: selectedFile.name,
          size: selectedFile.size,
          type: selectedFile.type
        }))
      } catch (err) {
        console.warn('Could not store file info:', err)
      }
    }
  }

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <nav className={`navbar ${isScrolled ? 'scrolled' : ''}`}>
      <div className="nav-container">
        <div className="nav-logo">
          <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '12px', textDecoration: 'none' }}>
            <div className="logo-container">
              <div className="waves">
                <div className="wave wave-1"></div>
                <div className="wave wave-2"></div>
                <div className="wave wave-3"></div>
                <div className="wave wave-4"></div>
                <div className="wave wave-5"></div>
              </div>
              <i className="fas fa-play"></i>
            </div>
            <span className="logo-text">F2P</span>
          </Link>
        </div>
        
        <ul className="nav-links">
          <li><Link to="/" className="nav-link">Home</Link></li>
          <li>
            <a 
              href="#features" 
              className="nav-link"
              onClick={(e) => {
                e.preventDefault()
                const element = document.getElementById('features')
                if (element) {
                  const offset = 100
                  const elementPosition = element.getBoundingClientRect().top + window.pageYOffset
                  const offsetPosition = elementPosition - offset
                  window.scrollTo({ top: offsetPosition, behavior: 'smooth' })
                }
              }}
            >
              Features
            </a>
          </li>
          <li>
            <a 
              href="#how-it-works" 
              className="nav-link"
              onClick={(e) => {
                e.preventDefault()
                const element = document.getElementById('how-it-works')
                if (element) {
                  const offset = 100
                  const elementPosition = element.getBoundingClientRect().top + window.pageYOffset
                  const offsetPosition = elementPosition - offset
                  window.scrollTo({ top: offsetPosition, behavior: 'smooth' })
                }
              }}
            >
              How It Works
            </a>
          </li>
          {isDashboardPage && (
            <li>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
              <button 
                onClick={handleUploadClick}
                className="btn-primary btn-small"
                style={{ cursor: 'pointer' }}
              >
                <i className="fas fa-upload" style={{ marginRight: '0.5rem' }} />
                Upload File
              </button>
            </li>
          )}
          <li><Link to="/dashboard" className="nav-link">Dashboard</Link></li>
        </ul>
      </div>
    </nav>
  )
}

export default Navbar

