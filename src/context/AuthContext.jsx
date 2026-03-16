import { createContext, useContext, useState, useEffect } from 'react'
import { signInWithPopup, signInWithRedirect, getRedirectResult, signOut, onAuthStateChanged } from 'firebase/auth'
import { auth, googleProvider } from '../lib/firebase'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Check for redirect result first
    // Try to resolve redirect sign-ins, but do NOT change `loading` here.
    // `onAuthStateChanged` will reliably fire and should control the loading flag
    getRedirectResult(auth)
      .then((result) => {
        if (result) {
          // User signed in via redirect
          setUser(result.user)
        }
      })
      .catch((error) => {
        console.error('Redirect result error:', error)
      })

    const unsubscribe = onAuthStateChanged(auth, (user) => {
      setUser(user)
      // Ensure loading is turned off once auth state is known
      setLoading(false)
    })

    return unsubscribe
  }, [])

  const signInWithGoogle = async () => {
    try {
      // Try popup first, fallback to redirect if COOP error
      try {
        const result = await signInWithPopup(auth, googleProvider)
        return result
      } catch (popupError) {
        // If popup fails due to COOP or other issues, use redirect
        const isCoopError = popupError.message?.includes?.('Cross-Origin-Opener-Policy') ||
                           popupError.code === 'auth/popup-blocked'
        
        if (popupError.code === 'auth/popup-closed-by-user' || isCoopError) {
          // COOP warnings are safe to ignore - redirect will handle authentication
          if (isCoopError) {
            console.log('COOP policy detected, using redirect authentication')
          } else {
            console.log('Popup blocked, using redirect instead')
          }
          await signInWithRedirect(auth, googleProvider)
          // Redirect will complete in the same window, so return null
          return null
        }
        throw popupError
      }
    } catch (error) {
      console.error('Error signing in with Google:', error)
      // Don't throw for redirect - it's expected
      if (error.code !== 'auth/cancelled-popup-request') {
        throw error
      }
    }
  }

  const logout = async () => {
    try {
      await signOut(auth)
    } catch (error) {
      console.error('Error signing out:', error)
      throw error
    }
  }

  const value = {
    user,
    loading,
    signInWithGoogle,
    logout
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

