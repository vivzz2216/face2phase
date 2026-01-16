"""
Cross-Validation System
Validates predictions between different analysis modules for consistency
"""
import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class CrossValidator:
    """Validates analysis results across modules for consistency"""
    
    def __init__(self):
        """Initialize cross-validator"""
        pass
    
    def validate_filler_detection(self, audio_results: Dict, text_results: Dict) -> Dict:
        """
        Cross-validate filler word detection between audio and text analysis
        
        Args:
            audio_results: Audio analysis results
            text_results: Text analysis results
            
        Returns:
            Validation results with agreement score
        """
        try:
            # Get filler counts from both sources
            audio_fillers = audio_results.get('filler_analysis', {}).get('total_fillers', 0)
            text_fillers = text_results.get('word_analysis', {}).get('filler_words', {}).get('filler_count', 0)
            
            # Calculate agreement
            if audio_fillers == 0 and text_fillers == 0:
                agreement = 1.0  # Perfect agreement (no fillers)
            elif audio_fillers == 0 or text_fillers == 0:
                agreement = 0.5  # One detected, other didn't
            else:
                # Calculate relative difference
                max_fillers = max(audio_fillers, text_fillers)
                min_fillers = min(audio_fillers, text_fillers)
                agreement = min_fillers / max_fillers if max_fillers > 0 else 0.0
            
            # Flag discrepancies
            discrepancy = abs(audio_fillers - text_fillers)
            has_discrepancy = discrepancy > max(audio_fillers, text_fillers) * 0.3  # >30% difference
            
            return {
                'audio_fillers': audio_fillers,
                'text_fillers': text_fillers,
                'agreement_score': float(agreement),
                'discrepancy': discrepancy,
                'has_discrepancy': has_discrepancy,
                'confidence': agreement
            }
            
        except Exception as e:
            logger.error(f"Error validating filler detection: {e}")
            return {
                'agreement_score': 0.5,
                'has_discrepancy': False,
                'confidence': 0.5
            }
    
    def validate_confidence_scores(self, audio_results: Dict, facial_results: Dict, text_results: Dict) -> Dict:
        """
        Validate consistency between confidence scores
        
        Args:
            audio_results: Audio analysis results
            facial_results: Facial analysis results
            text_results: Text analysis results
            
        Returns:
            Validation results
        """
        try:
            audio_score = audio_results.get('voice_confidence_score', 0)
            facial_score = facial_results.get('facial_confidence_score', 0)
            text_score = text_results.get('vocabulary_score', 0)
            
            scores = [s for s in [audio_score, facial_score, text_score] if s > 0]
            
            if len(scores) < 2:
                return {
                    'consistency_score': 0.5,
                    'has_inconsistency': False,
                    'confidence': 0.5
                }
            
            # Calculate variance
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv = std_score / mean_score if mean_score > 0 else 0  # Coefficient of variation
            
            # Lower CV = more consistent
            consistency_score = max(0.0, 1.0 - cv)
            
            # Flag if scores are very different (>30 points difference)
            score_range = max(scores) - min(scores)
            has_inconsistency = score_range > 30
            
            return {
                'scores': {
                    'audio': audio_score,
                    'facial': facial_score,
                    'text': text_score
                },
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'consistency_score': float(consistency_score),
                'score_range': float(score_range),
                'has_inconsistency': has_inconsistency,
                'confidence': consistency_score
            }
            
        except Exception as e:
            logger.error(f"Error validating confidence scores: {e}")
            return {
                'consistency_score': 0.5,
                'has_inconsistency': False,
                'confidence': 0.5
            }
    
    def validate_emotion_voice_consistency(self, facial_results: Dict, audio_results: Dict) -> Dict:
        """
        Validate consistency between facial emotions and voice tone
        
        Args:
            facial_results: Facial analysis results
            audio_results: Audio analysis results
            
        Returns:
            Validation results
        """
        try:
            # Get dominant emotion
            dominant_emotion = facial_results.get('dominant_emotion', 'neutral')
            
            # Get voice confidence (higher = more positive)
            voice_score = audio_results.get('voice_confidence_score', 50)
            
            # Map emotions to expected voice scores
            emotion_to_voice = {
                'happy': (70, 100),  # High voice confidence expected
                'neutral': (50, 80),
                'sad': (30, 60),  # Lower voice confidence
                'angry': (40, 70),
                'fear': (30, 60),
                'surprise': (60, 90),
                'disgust': (40, 70)
            }
            
            expected_range = emotion_to_voice.get(dominant_emotion, (40, 70))
            expected_min, expected_max = expected_range
            
            # Check if voice score is in expected range
            is_consistent = expected_min <= voice_score <= expected_max
            
            # Calculate consistency score
            if is_consistent:
                consistency_score = 1.0
            else:
                # Calculate distance from expected range
                if voice_score < expected_min:
                    distance = expected_min - voice_score
                    max_distance = expected_min - 0
                else:
                    distance = voice_score - expected_max
                    max_distance = 100 - expected_max
                
                consistency_score = max(0.0, 1.0 - (distance / max_distance) if max_distance > 0 else 0.0)
            
            return {
                'dominant_emotion': dominant_emotion,
                'voice_score': voice_score,
                'expected_range': expected_range,
                'is_consistent': is_consistent,
                'consistency_score': float(consistency_score),
                'confidence': consistency_score
            }
            
        except Exception as e:
            logger.error(f"Error validating emotion-voice consistency: {e}")
            return {
                'consistency_score': 0.5,
                'is_consistent': False,
                'confidence': 0.5
            }
    
    def validate_all(self, audio_results: Dict, facial_results: Dict, text_results: Dict, has_video: bool = False) -> Dict:
        """
        Perform all cross-validations
        
        Args:
            audio_results: Audio analysis results
            facial_results: Facial analysis results
            text_results: Text analysis results
            has_video: Whether video was analyzed
            
        Returns:
            Complete validation results
        """
        try:
            validations = {}
            
            # Validate filler detection
            validations['filler_validation'] = self.validate_filler_detection(audio_results, text_results)
            
            # Validate confidence scores
            validations['confidence_validation'] = self.validate_confidence_scores(
                audio_results, facial_results, text_results
            )
            
            # Validate emotion-voice consistency (if video)
            if has_video:
                validations['emotion_voice_validation'] = self.validate_emotion_voice_consistency(
                    facial_results, audio_results
                )
            
            # Calculate overall validation score
            validation_scores = [
                validations['filler_validation'].get('agreement_score', 0.5),
                validations['confidence_validation'].get('consistency_score', 0.5)
            ]
            
            if has_video:
                validation_scores.append(
                    validations['emotion_voice_validation'].get('consistency_score', 0.5)
                )
            
            overall_validation = np.mean(validation_scores)
            
            # Flag any major discrepancies
            has_issues = (
                validations['filler_validation'].get('has_discrepancy', False) or
                validations['confidence_validation'].get('has_inconsistency', False) or
                (has_video and not validations['emotion_voice_validation'].get('is_consistent', True))
            )
            
            validations['overall'] = {
                'validation_score': float(overall_validation),
                'has_issues': has_issues,
                'confidence': overall_validation
            }
            
            logger.info(f"Cross-validation completed: Overall score={overall_validation:.2f}, Issues={has_issues}")
            
            return validations
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {
                'overall': {
                    'validation_score': 0.5,
                    'has_issues': False,
                    'confidence': 0.5
                }
            }

# Global instance
cross_validator = CrossValidator()

