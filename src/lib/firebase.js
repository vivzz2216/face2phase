import { initializeApp } from "firebase/app"
import { getAuth, GoogleAuthProvider } from "firebase/auth"

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBi-6HOmBlYathM4-2KsaEIt0Ki6usm9gA",
  authDomain: "face2phase-b3d2c.firebaseapp.com",
  projectId: "face2phase-b3d2c",
  storageBucket: "face2phase-b3d2c.firebasestorage.app",
  messagingSenderId: "950391844283",
  appId: "1:950391844283:web:8320319ad693e28b6aa89e"
}

// Initialize Firebase
const app = initializeApp(firebaseConfig)

// Initialize Firebase Authentication and get a reference to the service
export const auth = getAuth(app)

// Configure Google Auth Provider
export const googleProvider = new GoogleAuthProvider()
googleProvider.setCustomParameters({
  prompt: 'consent',
  hd: 'google.com'
})
googleProvider.addScope('profile')
googleProvider.addScope('email')

export default app

