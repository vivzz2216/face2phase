"""
Tests for Cross Validator (Phase 3)
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cross_validator import CrossValidator


class TestCrossValidator:
    """Test suite for cross validator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.validator = CrossValidator()
    
    def test_validator_initialization(self):
        """Test that validator initializes correctly"""
        assert self.validator is not None
        assert hasattr(self.validator, 'validate_filler_detection')
        assert hasattr(self.validator, 'validate_confidence_scores')
        assert hasattr(self.validator, 'validate_emotion_voice_consistency')
        assert hasattr(self.validator, 'validate_all')
    
    def test_validate_filler_detection_agreement(self):
        """Test filler detection validation with agreement"""
        audio_results = {
            'filler_analysis': {
                'total_fillers': 5
            }
        }
        text_results = {
            'word_analysis': {
                'filler_words': {
                    'filler_count': 5
                }
            }
        }
        
        result = self.validator.validate_filler_detection(audio_results, text_results)
        
        assert result is not None
        assert 'agreement_score' in result
        assert 'audio_fillers' in result
        assert 'text_fillers' in result
        assert result['audio_fillers'] == 5
        assert result['text_fillers'] == 5
        assert result['agreement_score'] == 1.0  # Perfect agreement
        assert result['has_discrepancy'] == False
    
    def test_validate_filler_detection_disagreement(self):
        """Test filler detection validation with disagreement"""
        audio_results = {
            'filler_analysis': {
                'total_fillers': 10
            }
        }
        text_results = {
            'word_analysis': {
                'filler_words': {
                    'filler_count': 3
                }
            }
        }
        
        result = self.validator.validate_filler_detection(audio_results, text_results)
        
        assert result is not None
        assert result['audio_fillers'] == 10
        assert result['text_fillers'] == 3
        assert result['agreement_score'] < 1.0
        assert result['has_discrepancy'] == True
    
    def test_validate_filler_detection_no_fillers(self):
        """Test filler detection validation with no fillers"""
        audio_results = {
            'filler_analysis': {
                'total_fillers': 0
            }
        }
        text_results = {
            'word_analysis': {
                'filler_words': {
                    'filler_count': 0
                }
            }
        }
        
        result = self.validator.validate_filler_detection(audio_results, text_results)
        
        assert result['agreement_score'] == 1.0  # Perfect agreement (no fillers)
        assert result['has_discrepancy'] == False
    
    def test_validate_confidence_scores(self):
        """Test confidence score validation"""
        audio_results = {'voice_confidence_score': 75}
        facial_results = {'facial_confidence_score': 80}
        text_results = {'vocabulary_score': 70}
        
        result = self.validator.validate_confidence_scores(
            audio_results, facial_results, text_results
        )
        
        assert result is not None
        assert 'consistency_score' in result
        assert 'scores' in result
        assert result['scores']['audio'] == 75
        assert result['scores']['facial'] == 80
        assert result['scores']['text'] == 70
        assert 0 <= result['consistency_score'] <= 1.0
    
    def test_validate_confidence_scores_inconsistent(self):
        """Test confidence score validation with inconsistent scores"""
        audio_results = {'voice_confidence_score': 90}
        facial_results = {'facial_confidence_score': 30}
        text_results = {'vocabulary_score': 25}
        
        result = self.validator.validate_confidence_scores(
            audio_results, facial_results, text_results
        )
        
        assert result['has_inconsistency'] == True
        assert result['score_range'] > 30
    
    def test_validate_emotion_voice_consistency(self):
        """Test emotion-voice consistency validation"""
        facial_results = {'dominant_emotion': 'happy'}
        audio_results = {'voice_confidence_score': 75}
        
        result = self.validator.validate_emotion_voice_consistency(
            facial_results, audio_results
        )
        
        assert result is not None
        assert 'dominant_emotion' in result
        assert 'voice_score' in result
        assert 'is_consistent' in result
        assert 'consistency_score' in result
        assert result['dominant_emotion'] == 'happy'
        assert result['voice_score'] == 75
    
    def test_validate_all(self):
        """Test complete validation"""
        audio_results = {
            'filler_analysis': {'total_fillers': 5},
            'voice_confidence_score': 75
        }
        facial_results = {
            'facial_confidence_score': 80,
            'dominant_emotion': 'happy'
        }
        text_results = {
            'word_analysis': {
                'filler_words': {'filler_count': 5}
            },
            'vocabulary_score': 70
        }
        
        result = self.validator.validate_all(
            audio_results, facial_results, text_results, has_video=True
        )
        
        assert result is not None
        assert 'filler_validation' in result
        assert 'confidence_validation' in result
        assert 'emotion_voice_validation' in result
        assert 'overall' in result
        assert 'validation_score' in result['overall']
        assert 0 <= result['overall']['validation_score'] <= 1.0
    
    def test_validate_all_no_video(self):
        """Test validation without video"""
        audio_results = {
            'filler_analysis': {'total_fillers': 5},
            'voice_confidence_score': 75
        }
        facial_results = {}
        text_results = {
            'word_analysis': {
                'filler_words': {'filler_count': 5}
            },
            'vocabulary_score': 70
        }
        
        result = self.validator.validate_all(
            audio_results, facial_results, text_results, has_video=False
        )
        
        assert result is not None
        assert 'filler_validation' in result
        assert 'confidence_validation' in result
        assert 'emotion_voice_validation' not in result  # Should not be included
        assert 'overall' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

