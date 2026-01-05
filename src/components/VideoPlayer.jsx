import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import './VideoPlayer.css'

const VideoPlayer = ({ videoUrl, onTimeUpdate, currentTime }) => {
  const videoRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
    }

    const handleTimeUpdate = () => {
      if (onTimeUpdate) {
        onTimeUpdate(video.currentTime)
      }
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => setIsPlaying(false)

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('ended', handleEnded)

    // Sync with external currentTime prop
    if (currentTime !== undefined && Math.abs(video.currentTime - currentTime) > 0.5) {
      video.currentTime = currentTime
    }

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('ended', handleEnded)
    }
  }, [videoUrl, currentTime, onTimeUpdate])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play()
    }
  }

  const handleSeek = (e) => {
    const video = videoRef.current
    if (!video) return

    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const percent = x / rect.width
    video.currentTime = percent * duration
  }

  const handleVolumeChange = (e) => {
    const video = videoRef.current
    if (!video) return

    const newVolume = parseFloat(e.target.value)
    setVolume(newVolume)
    video.volume = newVolume
  }

  const handleSpeedChange = (speed) => {
    const video = videoRef.current
    if (!video) return

    setPlaybackRate(speed)
    video.playbackRate = speed
  }

  const toggleFullscreen = () => {
    const video = videoRef.current
    if (!video) return

    if (!isFullscreen) {
      if (video.requestFullscreen) {
        video.requestFullscreen()
      } else if (video.webkitRequestFullscreen) {
        video.webkitRequestFullscreen()
      } else if (video.mozRequestFullScreen) {
        video.mozRequestFullScreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen()
      } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen()
      }
    }
    setIsFullscreen(!isFullscreen)
  }

  const formatTime = (seconds) => {
    if (!seconds || isNaN(seconds)) return '0:00'
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const currentTimeValue = videoRef.current?.currentTime || 0

  return (
    <div className="video-player-container">
      <video
        ref={videoRef}
        src={videoUrl}
        className="video-element"
        onClick={togglePlay}
      />
      
      <div className="video-controls">
        <div className="controls-row">
          <button className="control-btn" onClick={togglePlay}>
            <i className={`fas ${isPlaying ? 'fa-pause' : 'fa-play'}`} />
          </button>

          <button 
            className="control-btn" 
            onClick={() => {
              const video = videoRef.current
              if (video) video.currentTime = Math.max(0, video.currentTime - 10)
            }}
          >
            <i className="fas fa-backward" />
          </button>

          <button 
            className="control-btn" 
            onClick={() => {
              const video = videoRef.current
              if (video) video.currentTime = Math.min(duration, video.currentTime + 10)
            }}
          >
            <i className="fas fa-forward" />
          </button>

          <div className="volume-control">
            <i className="fas fa-volume-up" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={handleVolumeChange}
              className="volume-slider"
            />
          </div>

          <div className="time-display">
            {formatTime(currentTimeValue)} / {formatTime(duration)}
          </div>

          <div className="speed-control">
            <select
              value={playbackRate}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              className="speed-select"
            >
              <option value={0.5}>0.5x</option>
              <option value={0.75}>0.75x</option>
              <option value={1}>1x</option>
              <option value={1.25}>1.25x</option>
              <option value={1.5}>1.5x</option>
              <option value={2}>2x</option>
            </select>
          </div>

          <button className="control-btn" onClick={toggleFullscreen}>
            <i className="fas fa-expand" />
          </button>
        </div>

        <div className="progress-bar-container" onClick={handleSeek}>
          <div 
            className="progress-bar"
            style={{ width: `${duration > 0 ? (currentTimeValue / duration) * 100 : 0}%` }}
          />
        </div>
      </div>
    </div>
  )
}

export default VideoPlayer

















