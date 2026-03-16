import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './context/AuthContext'
import Homepage from './components/Homepage'
import Dashboard from './components/Dashboard'
import VideoAnalysisPage from './components/VideoAnalysisPage'
import './App.css'

function App() {
  return (
    <AuthProvider>
      <Router
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true
        }}
      >
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/analysis/:sessionId" element={<VideoAnalysisPage />} />
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App

