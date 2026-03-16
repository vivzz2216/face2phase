/**
 * Audio Player Utility
 * Browser Text-to-Speech for pronunciation playback
 */

/**
 * Play pronunciation using browser SpeechSynthesis API
 * @param {string} text - Text to pronounce
 * @param {string} phonetic - IPA phonetic notation (optional, for display)
 * @param {string} language - Language code (default: 'en-US')
 * @returns {Promise<void>}
 */
export const playPronunciation = (text, phonetic = '', language = 'en-US') => {
  return new Promise((resolve, reject) => {
    if (!('speechSynthesis' in window)) {
      reject(new Error('Speech synthesis not supported in this browser'))
      return
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = language
    utterance.rate = 0.8 // Slightly slower for clarity
    utterance.pitch = 1.0
    utterance.volume = 1.0

    utterance.onend = () => {
      resolve()
    }

    utterance.onerror = (error) => {
      reject(error)
    }

    window.speechSynthesis.speak(utterance)
  })
}

/**
 * Stop any ongoing speech
 */
export const stopPronunciation = () => {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel()
  }
}

/**
 * Check if speech synthesis is available
 * @returns {boolean}
 */
export const isSpeechSynthesisAvailable = () => {
  return 'speechSynthesis' in window
}

/**
 * Get available voices
 * @returns {SpeechSynthesisVoice[]}
 */
export const getAvailableVoices = () => {
  if (!('speechSynthesis' in window)) {
    return []
  }
  return window.speechSynthesis.getVoices()
}

/**
 * Play pronunciation with specific voice
 * @param {string} text - Text to pronounce
 * @param {SpeechSynthesisVoice} voice - Voice to use
 * @returns {Promise<void>}
 */
export const playPronunciationWithVoice = (text, voice) => {
  return new Promise((resolve, reject) => {
    if (!('speechSynthesis' in window)) {
      reject(new Error('Speech synthesis not supported'))
      return
    }

    window.speechSynthesis.cancel()

    const utterance = new SpeechSynthesisUtterance(text)
    if (voice) {
      utterance.voice = voice
    }
    utterance.lang = voice?.lang || 'en-US'
    utterance.rate = 0.8
    utterance.pitch = 1.0
    utterance.volume = 1.0

    utterance.onend = () => resolve()
    utterance.onerror = (error) => reject(error)

    window.speechSynthesis.speak(utterance)
  })
}















