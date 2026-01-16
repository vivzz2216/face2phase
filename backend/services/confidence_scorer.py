"""
Confidence Scoring System
Provides confidence scores for all analysis predictions
"""
import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """Calculates confidence scores for analysis predictions"""
    
    def __init__(self):
        """Initialize confidence scorer"""
        pass
    
    def calculate_audio_confidence(self, audio_results: Dict) -> float:
        """
        Calculate confidence score for audio analysis
        
        Args:
            audio_results: Audio analysis results
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence_factors = []
            
            # Transcription confidence (from Whisper)
            words_with_timing = audio_results.get('words_with_timing', [])
            if words_with_timing:
                avg_probability = np.mean([w.get('probability', 0.5) for w in words_with_timing])
                confidence_factors.append(avg_probability)
            else:
                confidence_factors.append(0.3)  # Low confidence if no words
            
            # Audio quality indicators
            transcript = audio_results.get('transcript', '')
            if len(transcript) > 10:
                confidence_factors.append(0.8)  # Good if we got transcript
            else:
                confidence_factors.append(0.3)
            
            # Pause detection quality
            pauses = audio_results.get('pauses', [])
            if len(pauses) > 0:
                confidence_factors.append(0.7)  # Pauses detected = good audio quality
            else:
                confidence_factors.append(0.5)  # Neutral
            
            # Filler detection quality
            filler_analysis = audio_results.get('filler_analysis', {})
            if filler_analysis.get('total_fillers', 0) >= 0:
                confidence_factors.append(0.7)  # Analysis completed
            
            # Weighted average
            weights = [0.4, 0.3, 0.15, 0.15]  # Transcription most important
            confidence = np.average(confidence_factors[:len(weights)], weights=weights[:len(confidence_factors)])
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating audio confidence: {e}")
            return 0.5  # Default medium confidence
    
    def calculate_facial_confidence(self, facial_results: Dict) -> float:
        """
        Calculate confidence score for facial analysis
        
        Args:
            facial_results: Facial analysis results
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence_factors = []
            
            # Face detection rate
            face_detection_rate = facial_results.get('face_detection_rate', 0)
            confidence_factors.append(face_detection_rate)
            
            # Number of frames analyzed
            frame_results = facial_results.get('frame_results', [])
            if len(frame_results) > 10:
                confidence_factors.append(0.8)  # Good sample size
            elif len(frame_results) > 5:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.4)  # Small sample
            
            # Emotion detection consistency
            emotions = [r.get('dominant_emotion', 'neutral') for r in frame_results]
            if len(emotions) > 0:
                unique_emotions = len(set(emotions))
                emotion_consistency = 1.0 - (unique_emotions / len(emotions) * 0.3)  # More consistent = higher confidence
                confidence_factors.append(emotion_consistency)
            
            # Eye contact detection
            eye_scores = [r.get('eye_contact_score', 0.5) for r in frame_results]
            if eye_scores:
                avg_eye_contact = np.mean(eye_scores)
                # Higher eye contact = more reliable detection
                confidence_factors.append(min(1.0, avg_eye_contact * 1.2))
            
            # Weighted average
            if len(confidence_factors) > 0:
                weights = [0.3, 0.25, 0.25, 0.2][:len(confidence_factors)]
                confidence = np.average(confidence_factors, weights=weights)
            else:
                confidence = 0.5
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating facial confidence: {e}")
            return 0.5
    
    def calculate_text_confidence(self, text_results: Dict) -> float:
        """
        Calculate confidence score for text analysis
        
        Args:
            text_results: Text analysis results
            
        Returns:
            Confidence score (0-1)
        """
        try:
            confidence_factors = []
            
            # Vocabulary metrics availability
            vocab_metrics = text_results.get('vocabulary_metrics', {})
            if vocab_metrics:
                confidence_factors.append(0.8)  # Metrics available
            else:
                confidence_factors.append(0.4)
            
            # Text length (more text = more reliable)
            total_words = vocab_metrics.get('total_words', 0)
            if total_words > 100:
                confidence_factors.append(0.9)
            elif total_words > 50:
                confidence_factors.append(0.7)
            elif total_words > 20:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)
            
            # Analysis success flag
            if text_results.get('analysis_successful', False):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)
            
            # Average confidence
            if len(confidence_factors) > 0:
                confidence = np.mean(confidence_factors)
            else:
                confidence = 0.5
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating text confidence: {e}")
            return 0.5
    
    def calculate_overall_confidence(self, audio_confidence: float, facial_confidence: float, text_confidence: float, has_video: bool = False) -> float:
        """
        Calculate overall confidence score
        
        Args:
            audio_confidence: Audio analysis confidence
            facial_confidence: Facial analysis confidence
            text_confidence: Text analysis confidence
            has_video: Whether video was analyzed
            
        Returns:
            Overall confidence score (0-1)
        """
        try:
            if has_video:
                # For video: weight all three equally
                weights = [0.4, 0.3, 0.3]  # Audio, Facial, Text
                confidences = [audio_confidence, facial_confidence, text_confidence]
            else:
                # For audio-only: weight audio and text
                weights = [0.6, 0.4]  # Audio, Text
                confidences = [audio_confidence, text_confidence]
            
            overall = np.average(confidences, weights=weights)
            
            return float(np.clip(overall, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.5
    
    def add_confidence_scores(self, results: Dict, has_video: bool = False) -> Dict:
        """
        Add confidence scores to analysis results
        
        Args:
            results: Analysis results dictionary
            has_video: Whether video was analyzed
            
        Returns:
            Results with confidence scores added
        """
        try:
            # Calculate individual confidences
            audio_conf = self.calculate_audio_confidence(results.get('audio_results', {}))
            facial_conf = self.calculate_facial_confidence(results.get('facial_results', {})) if has_video else 0.0
            text_conf = self.calculate_text_confidence(results.get('text_results', {}))
            
            # Calculate overall
            overall_conf = self.calculate_overall_confidence(
                audio_conf, facial_conf, text_conf, has_video
            )
            
            # Add to results
            results['confidence_scores'] = {
                'audio': round(audio_conf, 3),
                'facial': round(facial_conf, 3) if has_video else None,
                'text': round(text_conf, 3),
                'overall': round(overall_conf, 3)
            }
            
            logger.info(f"Confidence scores: Audio={audio_conf:.2f}, Facial={facial_conf:.2f}, Text={text_conf:.2f}, Overall={overall_conf:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error adding confidence scores: {e}")
            results['confidence_scores'] = {
                'audio': 0.5,
                'facial': 0.5 if has_video else None,
                'text': 0.5,
                'overall': 0.5
            }
            return results

# Global instance
confidence_scorer = ConfidenceScorer()

