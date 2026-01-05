"""
Tests for Acoustic Pronunciation Analyzer (Phase 3)
"""
import pytest
import numpy as np
import librosa
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.acoustic_pronunciation_analyzer import AcousticPronunciationAnalyzer


class TestAcousticPronunciationAnalyzer:
    """Test suite for acoustic pronunciation analyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = AcousticPronunciationAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'extract_acoustic_features')
        assert hasattr(self.analyzer, 'compare_pronunciation')
        assert hasattr(self.analyzer, 'detect_stress_from_audio')
    
    def test_extract_acoustic_features(self):
        """Test acoustic feature extraction"""
        # Create synthetic audio signal (1 second of sine wave)
        sr = 16000
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration))
        audio_segment = np.sin(2 * np.pi * frequency * t)
        
        features = self.analyzer.extract_acoustic_features(audio_segment, sr)
        
        # Check that features were extracted
        assert features is not None
        assert isinstance(features, dict)
        
        # Check for expected feature keys
        expected_keys = ['mfcc_mean', 'spectral_centroid_mean', 'pitch_mean']
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"
    
    def test_compare_pronunciation_no_reference(self):
        """Test pronunciation comparison without reference"""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio_segment = np.sin(2 * np.pi * 440 * t)
        
        result = self.analyzer.compare_pronunciation(audio_segment, sr)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'similarity_score' in result
        assert 'pronunciation_accuracy' in result
        assert 'confidence' in result
        assert 'method' in result
        
        # Should use quality estimation when no reference
        assert result['method'] in ['quality_estimation', 'acoustic_comparison', 'error']
        assert 0 <= result['pronunciation_accuracy'] <= 100
    
    def test_compare_pronunciation_with_reference(self):
        """Test pronunciation comparison with reference"""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio_segment = np.sin(2 * np.pi * 440 * t)
        
        # Create reference features
        reference_features = {
            'mfcc_mean': [0.0] * 13,
            'spectral_centroid_mean': 2000.0,
            'pitch_mean': 440.0
        }
        
        result = self.analyzer.compare_pronunciation(
            audio_segment, sr, reference_features=reference_features
        )
        
        assert result is not None
        assert 'similarity_score' in result
        assert 'pronunciation_accuracy' in result
        assert result['method'] == 'acoustic_comparison'
        assert 0 <= result['pronunciation_accuracy'] <= 100
    
    def test_detect_stress_from_audio(self):
        """Test stress detection from audio"""
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create audio with varying pitch (simulating stress)
        audio_segment = np.sin(2 * np.pi * (440 + 100 * t) * t)
        
        result = self.analyzer.detect_stress_from_audio(audio_segment, sr)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'stress_detected' in result
        assert 'stress_score' in result
        assert 'confidence' in result
        assert isinstance(result['stress_detected'], bool)
        assert 0 <= result['stress_score'] <= 1.0
    
    def test_get_reference_features(self):
        """Test getting reference features"""
        # Should return None for words not in cache (expected behavior)
        result = self.analyzer.get_reference_features("testword")
        assert result is None or isinstance(result, dict)
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        # Test with empty audio
        result = self.analyzer.extract_acoustic_features(np.array([]), 16000)
        assert result is not None  # Should handle gracefully
        
        # Test with invalid sample rate
        audio = np.array([0.1, 0.2, 0.3])
        result = self.analyzer.extract_acoustic_features(audio, 0)
        assert result is not None  # Should handle gracefully


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

