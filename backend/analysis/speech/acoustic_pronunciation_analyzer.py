"""
Acoustic Pronunciation Analysis
Compares actual audio pronunciation with reference pronunciations using acoustic features
"""
import librosa
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class AcousticPronunciationAnalyzer:
    """Analyzes pronunciation by comparing acoustic features with reference"""
    
    def __init__(self):
        """Initialize acoustic pronunciation analyzer"""
        # Reference pronunciation database (can be expanded)
        self.reference_features_cache = {}
    
    def extract_acoustic_features(self, audio_segment: np.ndarray, sr: int) -> Dict:
        """
        Extract acoustic features from audio segment
        
        Args:
            audio_segment: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of acoustic features
        """
        try:
            features = {}
            
            # MFCC (Mel-frequency cepstral coefficients) - captures spectral envelope
            mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            # Spectral centroid - brightness of sound
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # Spectral rolloff - frequency below which 85% of energy is contained
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Zero-crossing rate - measure of noisiness
            zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
            features['zcr_mean'] = float(np.mean(zcr))
            
            # Chroma - pitch class distribution
            try:
                chroma = librosa.feature.chroma_stft(y=audio_segment, sr=sr)
                features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            except AttributeError:
                # Fallback if chroma_stft not available
                try:
                    chroma = librosa.feature.chroma(y=audio_segment, sr=sr)
                    features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
                except:
                    features['chroma_mean'] = [0.0] * 12  # Default 12 chroma bins
            
            # Pitch (fundamental frequency)
            try:
                pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                else:
                    features['pitch_mean'] = 0.0
                    features['pitch_std'] = 0.0
            except:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            return {}
    
    def get_reference_features(self, word: str, phonemes: Optional[List[str]] = None) -> Optional[Dict]:
        """
        Get reference acoustic features for a word
        
        Args:
            word: Word to get reference for
            phonemes: Optional phoneme sequence from CMU dict
            
        Returns:
            Reference features dictionary or None
        """
        # For now, use cached features or generate from TTS
        # In production, this would use a database of reference pronunciations
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        
        if word_lower in self.reference_features_cache:
            return self.reference_features_cache[word_lower]
        
        # Could generate reference using TTS here
        # For now, return None (will use fallback methods)
        return None
    
    def compare_pronunciation(self, audio_segment: np.ndarray, sr: int, 
                             reference_features: Optional[Dict] = None,
                             word: Optional[str] = None) -> Dict:
        """
        Compare actual pronunciation with reference
        
        Args:
            audio_segment: Audio segment to analyze
            sr: Sample rate
            reference_features: Optional reference features
            word: Optional word for context
            
        Returns:
            Comparison results with similarity score
        """
        try:
            # Extract features from actual audio
            actual_features = self.extract_acoustic_features(audio_segment, sr)
            
            if not actual_features:
                return {
                    'similarity_score': 0.0,
                    'pronunciation_accuracy': 0.0,
                    'confidence': 0.0,
                    'method': 'feature_extraction_failed'
                }
            
            # If reference available, compare
            if reference_features:
                # Calculate similarity for each feature type
                similarities = []
                
                # MFCC similarity (cosine distance)
                if 'mfcc_mean' in actual_features and 'mfcc_mean' in reference_features:
                    actual_mfcc = np.array(actual_features['mfcc_mean'])
                    ref_mfcc = np.array(reference_features['mfcc_mean'])
                    if np.linalg.norm(actual_mfcc) > 0 and np.linalg.norm(ref_mfcc) > 0:
                        mfcc_sim = 1 - cosine(actual_mfcc, ref_mfcc)
                        similarities.append(max(0, mfcc_sim))
                
                # Spectral centroid similarity
                if 'spectral_centroid_mean' in actual_features and 'spectral_centroid_mean' in reference_features:
                    actual_sc = actual_features['spectral_centroid_mean']
                    ref_sc = reference_features['spectral_centroid_mean']
                    if ref_sc > 0:
                        sc_sim = 1 - abs(actual_sc - ref_sc) / max(actual_sc, ref_sc, 1)
                        similarities.append(max(0, sc_sim))
                
                # Pitch similarity
                if 'pitch_mean' in actual_features and 'pitch_mean' in reference_features:
                    actual_pitch = actual_features['pitch_mean']
                    ref_pitch = reference_features['pitch_mean']
                    if ref_pitch > 0 and actual_pitch > 0:
                        pitch_sim = 1 - abs(actual_pitch - ref_pitch) / max(actual_pitch, ref_pitch)
                        similarities.append(max(0, pitch_sim))
                
                # Overall similarity
                if similarities:
                    similarity_score = np.mean(similarities)
                    pronunciation_accuracy = similarity_score * 100
                else:
                    similarity_score = 0.5
                    pronunciation_accuracy = 50.0
                
                return {
                    'similarity_score': float(similarity_score),
                    'pronunciation_accuracy': float(pronunciation_accuracy),
                    'confidence': 0.8 if len(similarities) >= 2 else 0.5,
                    'method': 'acoustic_comparison',
                    'feature_similarities': {
                        'mfcc': similarities[0] if len(similarities) > 0 else None,
                        'spectral': similarities[1] if len(similarities) > 1 else None,
                        'pitch': similarities[2] if len(similarities) > 2 else None
                    }
                }
            else:
                # No reference - use quality metrics
                # Higher quality = likely better pronunciation
                quality_score = 0.5
                
                # Check if features indicate clear speech
                if actual_features.get('spectral_centroid_mean', 0) > 1000:
                    quality_score += 0.1  # Bright sound = clear
                
                if actual_features.get('pitch_mean', 0) > 100:
                    quality_score += 0.1  # Has pitch = voiced
                
                if actual_features.get('zcr_mean', 0) < 0.1:
                    quality_score += 0.1  # Low ZCR = less noisy
                
                quality_score = min(1.0, quality_score)
                
                return {
                    'similarity_score': quality_score,
                    'pronunciation_accuracy': quality_score * 100,
                    'confidence': 0.4,  # Lower confidence without reference
                    'method': 'quality_estimation',
                    'quality_indicators': {
                        'spectral_centroid': actual_features.get('spectral_centroid_mean', 0),
                        'pitch': actual_features.get('pitch_mean', 0),
                        'zcr': actual_features.get('zcr_mean', 0)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error comparing pronunciation: {e}")
            return {
                'similarity_score': 0.0,
                'pronunciation_accuracy': 0.0,
                'confidence': 0.0,
                'method': 'error',
                'error': str(e)
            }
    
    def detect_stress_from_audio(self, audio_segment: np.ndarray, sr: int) -> Dict:
        """
        Detect stress pattern from audio using pitch and energy
        
        Args:
            audio_segment: Audio segment
            sr: Sample rate
            
        Returns:
            Stress detection results
        """
        try:
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            # Extract energy
            rms = librosa.feature.rms(y=audio_segment)[0]
            energy_values = rms.tolist()
            
            if not pitch_values or not energy_values:
                return {
                    'stress_detected': False,
                    'stress_score': 0.0,
                    'confidence': 0.0
                }
            
            # Normalize pitch and energy
            pitch_norm = (np.array(pitch_values) - np.min(pitch_values)) / (np.max(pitch_values) - np.min(pitch_values) + 1e-10)
            energy_norm = (np.array(energy_values) - np.min(energy_values)) / (np.max(energy_values) - np.min(energy_values) + 1e-10)
            
            # Stressed syllables: higher pitch + higher energy
            # Average the normalized values
            if len(pitch_norm) == len(energy_norm):
                stress_scores = (pitch_norm + energy_norm) / 2
            else:
                # Interpolate to match lengths
                min_len = min(len(pitch_norm), len(energy_norm))
                stress_scores = (pitch_norm[:min_len] + energy_norm[:min_len]) / 2
            
            # Find peak stress
            max_stress_idx = np.argmax(stress_scores) if len(stress_scores) > 0 else 0
            max_stress = stress_scores[max_stress_idx] if len(stress_scores) > 0 else 0.0
            
            # Threshold for stress detection
            stress_threshold = 0.6
            stress_detected = max_stress > stress_threshold
            
            return {
                'stress_detected': bool(stress_detected),
                'stress_score': float(max_stress),
                'stress_position': int(max_stress_idx),
                'confidence': float(min(1.0, max_stress * 1.2)),
                'all_stress_scores': stress_scores.tolist() if len(stress_scores) < 20 else stress_scores[:20].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error detecting stress from audio: {e}")
            return {
                'stress_detected': False,
                'stress_score': 0.0,
                'confidence': 0.0
            }

# Global instance
acoustic_pronunciation_analyzer = AcousticPronunciationAnalyzer()

