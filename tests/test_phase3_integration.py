"""
Integration tests for Phase 3 features
Tests that all Phase 3 components work together
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.acoustic_pronunciation_analyzer import acoustic_pronunciation_analyzer
from models.cross_validator import cross_validator
from models.ml_pattern_recognizer import ml_pattern_recognizer
from models.word_analyzer import WordAnalyzer


class TestPhase3Integration:
    """Integration tests for Phase 3 features"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.word_analyzer = WordAnalyzer()
    
    def test_cross_validator_import(self):
        """Test that cross validator can be imported"""
        assert cross_validator is not None
        assert hasattr(cross_validator, 'validate_all')
    
    def test_acoustic_analyzer_import(self):
        """Test that acoustic analyzer can be imported"""
        assert acoustic_pronunciation_analyzer is not None
        assert hasattr(acoustic_pronunciation_analyzer, 'extract_acoustic_features')
    
    def test_ml_recognizer_import(self):
        """Test that ML recognizer can be imported"""
        assert ml_pattern_recognizer is not None
        assert hasattr(ml_pattern_recognizer, 'get_weak_word_patterns')
    
    def test_word_analyzer_with_ml_integration(self):
        """Test that word analyzer can use ML pattern recognition"""
        transcript = "I just really want to say, like, you know, it's pretty good."
        
        result = self.word_analyzer.analyze_weak_words(transcript)
        
        assert result is not None
        assert 'weak_words_found' in result
        assert 'weak_word_count' in result
        assert isinstance(result['weak_words_found'], list)
        # ML integration should work (may use fallback if ML not trained)
    
    def test_cross_validation_workflow(self):
        """Test complete cross-validation workflow"""
        # Simulate analysis results
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
        
        # Run cross-validation
        validation = cross_validator.validate_all(
            audio_results, facial_results, text_results, has_video=True
        )
        
        assert validation is not None
        assert 'overall' in validation
        assert 'validation_score' in validation['overall']
    
    def test_all_modules_available(self):
        """Test that all Phase 3 modules are available"""
        modules = [
            ('acoustic_pronunciation_analyzer', acoustic_pronunciation_analyzer),
            ('cross_validator', cross_validator),
            ('ml_pattern_recognizer', ml_pattern_recognizer),
        ]
        
        for name, module in modules:
            assert module is not None, f"{name} module not available"
    
    def test_error_handling_integration(self):
        """Test error handling across integrated modules"""
        # Test with empty/invalid data
        empty_transcript = ""
        result = self.word_analyzer.analyze_weak_words(empty_transcript)
        
        assert result is not None
        assert result['weak_word_count'] == 0
        
        # Test cross-validation with missing data
        empty_audio = {}
        empty_facial = {}
        empty_text = {}
        
        validation = cross_validator.validate_all(
            empty_audio, empty_facial, empty_text, has_video=False
        )
        
        assert validation is not None
        assert 'overall' in validation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

