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
  const userIsSeekingRef = useRef(false)  // Track if user is manually seeking

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

    const handleSeeking = () => {
      console.log('🔍 Video seeking event - currentTime:', video.currentTime, 'userIsSeeking:', userIsSeekingRef.current)
    }

    const handleSeeked = () => {
      console.log('✅ Video seeked event - final currentTime:', video.currentTime)
      // Reset the flag after a short delay to allow sync again
      setTimeout(() => {
        console.log('🔓 Unlocking userIsSeekingRef')
        userIsSeekingRef.current = false
      }, 100)
    }

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)
    video.addEventListener('ended', handleEnded)
    video.addEventListener('seeking', handleSeeking)
    video.addEventListener('seeked', handleSeeked)

    // Sync with external currentTime prop ONLY if user is not manually seeking
    if (!userIsSeekingRef.current && currentTime !== undefined && Math.abs(video.currentTime - currentTime) > 0.5) {
      console.log('🔄 Syncing to external currentTime:', currentTime, 'from:', video.currentTime, 'userIsSeeking:', userIsSeekingRef.current)
      video.currentTime = currentTime
    } else if (userIsSeekingRef.current) {
      console.log('🚫 BLOCKED sync because userIsSeeking is true. currentTime prop:', currentTime, 'video.currentTime:', video.currentTime)
    }

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
      video.removeEventListener('ended', handleEnded)
      video.removeEventListener('seeking', handleSeeking)
      video.removeEventListener('seeked', handleSeeked)
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
    const percent = Math.max(0, Math.min(1, x / rect.width))
    video.currentTime = percent * duration
  }

  const handleProgressMouseDown = (e) => {
    const video = videoRef.current
    if (!video || !video.duration) {
      console.log('Video not ready or duration not available')
      return
    }

    // Mark that user is manually seeking - prevent useEffect from interfering
    userIsSeekingRef.current = true

    // Prevent default to avoid any conflicts
    e.preventDefault()
    e.stopPropagation()

    // Store the progress bar element reference
    const progressBar = e.currentTarget
    const videoDuration = video.duration

    const calculateAndSeek = (clientX) => {
      const rect = progressBar.getBoundingClientRect()
      const x = Math.max(0, clientX - rect.left)
      const percent = Math.max(0, Math.min(1, x / rect.width))
      const newTime = percent * videoDuration

      console.log('Seeking:', {
        x,
        width: rect.width,
        percent: (percent * 100).toFixed(1) + '%',
        newTime: newTime.toFixed(2) + 's',
        duration: videoDuration.toFixed(2) + 's'
      })

      // Only seek if the time is valid
      if (isFinite(newTime) && newTime >= 0 && newTime <= videoDuration) {
        video.currentTime = newTime
      }
    }

    const handleMouseMove = (moveEvent) => {
      calculateAndSeek(moveEvent.clientX)
    }

    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      // Flag will be reset by 'seeked' event listener
    }

    // Seek immediately on mouse down
    calculateAndSeek(e.clientX)

    // Listen for drag
    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
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
              if (video) {
                userIsSeekingRef.current = true
                video.currentTime = Math.max(0, video.currentTime - 10)
              }
            }}
          >
            <i className="fas fa-backward" />
          </button>

          <button
            className="control-btn"
            onClick={() => {
              const video = videoRef.current
              if (video) {
                userIsSeekingRef.current = true
                video.currentTime = Math.min(duration, video.currentTime + 10)
              }
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

        <div
          className="progress-bar-container"
          onMouseDown={handleProgressMouseDown}
          style={{ cursor: 'pointer' }}
        >
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

















