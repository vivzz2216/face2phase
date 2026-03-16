"""
Data Validation Utilities for Speech Analysis
Ensures all metrics are mathematically correct and within valid ranges
"""
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when a metric fails validation"""
    pass


class MetricValidator:
    """Validates all speech analysis metrics for accuracy and consistency"""
    
    # Valid ranges for different metric types
    METRIC_RANGES = {
        'percentage': (0.0, 100.0),
        'score_100': (0.0, 100.0),
        'wpm': (30.0, 300.0),  # Physically plausible speaking rate
        'duration': (0.0, 86400.0),  # Max 24 hours
        'count': (0, float('inf')),
    }
    
    # Confidence thresholds
    MIN_TRANSCRIPTION_CONFIDENCE = 0.70
    MIN_METRIC_CONFIDENCE = 0.65
    
    @staticmethod
    def validate_percentage(value: float, metric_name: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate that a value is a valid percentage (0-100)
        
        Args:
            value: The value to validate
            metric_name: Name of the metric for error messages
            strict: If True, raises error; if False, returns validation result
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            return (False, f"{metric_name}: Value is None")
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            return (False, f"{metric_name}: Cannot convert to float: {value}")
        
        if value < 0:
            error = f"{metric_name}: Percentage cannot be negative: {value}"
            if strict:
                logger.error(error)
            return (False, error)
        
        if value > 100:
            error = f"{metric_name}: Percentage cannot exceed 100: {value}"
            if strict:
                logger.error(error)
            return (False, error)
        
        return (True, None)
    
    @staticmethod
    def validate_score(value: float, min_val: float, max_val: float, 
                       metric_name: str, strict: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate that a score is within expected range
        
        Args:
            value: The score to validate
            min_val: Minimum valid value
            max_val: Maximum valid value
            metric_name: Name of the metric
            strict: If True, logs errors
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None:
            return (False, f"{metric_name}: Value is None")
        
        try:
            value = float(value)
        except (TypeError, ValueError):
            return (False, f"{metric_name}: Cannot convert to float: {value}")
        
        if value < min_val:
            error = f"{metric_name}: Value {value} below minimum {min_val}"
            if strict:
                logger.error(error)
            return (False, error)
        
        if value > max_val:
            error = f"{metric_name}: Value {value} exceeds maximum {max_val}"
            if strict:
                logger.error(error)
            return (False, error)
        
        return (True, None)
    
    @staticmethod
    def validate_math_precision(calculated: float, expected: float, 
                                 tolerance: float = 0.01, 
                                 metric_name: str = "Metric") -> Tuple[bool, Optional[str]]:
        """
        Validate that a calculated value matches expected with given tolerance
        
        Args:
            calculated: The calculated value
            expected: The expected value
            tolerance: Acceptable difference
            metric_name: Name for error messages
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if calculated is None or expected is None:
            return (False, f"{metric_name}: Cannot validate None values")
        
        try:
            diff = abs(float(calculated) - float(expected))
        except (TypeError, ValueError):
            return (False, f"{metric_name}: Cannot calculate difference")
        
        if diff > tolerance:
            error = f"{metric_name}: Calculation mismatch - got {calculated}, expected {expected} (diff: {diff})"
            logger.error(error)
            return (False, error)
        
        return (True, None)
    
    @staticmethod
    def validate_filler_calculation(filler_count: int, total_words: int, 
                                     reported_percentage: float) -> Tuple[bool, float, Optional[str]]:
        """
        Validate filler word percentage calculation
        
        Formula: (filler_count / total_words) * 100
        
        Args:
            filler_count: Number of filler words detected
            total_words: Total word count
            reported_percentage: The percentage being reported
            
        Returns:
            Tuple of (is_valid, correct_percentage, error_message)
        """
        if total_words <= 0:
            if filler_count > 0:
                return (False, 0.0, "Cannot have fillers with zero total words")
            return (True, 0.0, None)
        
        if filler_count < 0:
            return (False, 0.0, "Filler count cannot be negative")
        
        if filler_count > total_words:
            return (False, 0.0, f"Filler count ({filler_count}) exceeds total words ({total_words})")
        
        # Calculate correct percentage
        correct_percentage = round((filler_count / total_words) * 100, 2)
        
        # Check if reported matches calculated
        tolerance = 0.1  # Allow 0.1% tolerance for rounding
        if abs(reported_percentage - correct_percentage) > tolerance:
            error = f"Filler percentage mismatch: reported {reported_percentage}%, calculated {correct_percentage}%"
            logger.warning(error)
            return (False, correct_percentage, error)
        
        return (True, correct_percentage, None)
    
    @staticmethod
    def validate_wpm_calculation(word_count: int, duration_seconds: float, 
                                  reported_wpm: float) -> Tuple[bool, float, Optional[str]]:
        """
        Validate Words Per Minute calculation
        
        Formula: word_count / (duration_seconds / 60)
        
        Args:
            word_count: Number of words
            duration_seconds: Duration in seconds
            reported_wpm: The WPM being reported
            
        Returns:
            Tuple of (is_valid, correct_wpm, error_message)
        """
        if duration_seconds <= 0:
            return (False, 0.0, "Duration must be positive")
        
        if word_count < 0:
            return (False, 0.0, "Word count cannot be negative")
        
        # Calculate correct WPM
        correct_wpm = round(word_count / (duration_seconds / 60), 1)
        
        # Validate physically plausible range
        if correct_wpm > 300:
            logger.warning(f"WPM {correct_wpm} exceeds human speech rate - may indicate error")
        elif correct_wpm < 30 and word_count > 0:
            logger.warning(f"WPM {correct_wpm} unusually slow - may indicate error")
        
        # Check if reported matches calculated
        tolerance = 1.0  # Allow 1 WPM tolerance
        if abs(reported_wpm - correct_wpm) > tolerance:
            error = f"WPM mismatch: reported {reported_wpm}, calculated {correct_wpm}"
            logger.warning(error)
            return (False, correct_wpm, error)
        
        return (True, correct_wpm, None)
    
    @staticmethod
    def validate_pause_analysis(pauses: List[Dict], total_duration: float) -> Tuple[bool, Optional[str]]:
        """
        Validate pause analysis data
        
        Args:
            pauses: List of pause dictionaries with start, end, duration
            total_duration: Total audio duration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not pauses:
            return (True, None)
        
        total_pause_time = 0.0
        
        for i, pause in enumerate(pauses):
            start = pause.get('start', 0)
            end = pause.get('end', 0)
            duration = pause.get('duration', 0)
            
            # Validate individual pause
            if start < 0:
                return (False, f"Pause {i}: Start time cannot be negative")
            
            if end < start:
                return (False, f"Pause {i}: End time before start time")
            
            if duration < 0:
                return (False, f"Pause {i}: Duration cannot be negative")
            
            # Check duration calculation
            expected_duration = end - start
            if abs(duration - expected_duration) > 0.1:
                return (False, f"Pause {i}: Duration mismatch - reported {duration}, calculated {expected_duration}")
            
            if end > total_duration:
                return (False, f"Pause {i}: End time exceeds total duration")
            
            total_pause_time += duration
        
        # Total pause time should not exceed total duration
        if total_pause_time > total_duration:
            return (False, f"Total pause time ({total_pause_time}s) exceeds audio duration ({total_duration}s)")
        
        return (True, None)
    
    @staticmethod
    def detect_impossible_values(metrics: Dict[str, Any]) -> List[str]:
        """
        Detect physically impossible or contradictory metric combinations
        
        Args:
            metrics: Dictionary of analysis metrics
            
        Returns:
            List of error messages for impossible values
        """
        errors = []
        
        # Check for percentages > 100
        percentage_keys = [
            'filler_percentage', 'eye_contact_percentage', 'silence_percentage',
            'weak_word_percentage', 'vocabulary_richness'
        ]
        for key in percentage_keys:
            if key in metrics:
                value = metrics.get(key)
                if value is not None:
                    try:
                        val = float(value)
                        if val < 0 or val > 100:
                            errors.append(f"{key}: {val} is outside valid range [0, 100]")
                    except (TypeError, ValueError):
                        errors.append(f"{key}: Cannot convert to number")
        
        # Check for negative counts
        count_keys = ['word_count', 'filler_count', 'pause_count', 'weak_word_count']
        for key in count_keys:
            if key in metrics:
                value = metrics.get(key)
                if value is not None and value < 0:
                    errors.append(f"{key}: Cannot be negative")
        
        # Check for impossible combinations
        filler_count = metrics.get('filler_count', 0)
        word_count = metrics.get('word_count', 0)
        if filler_count > word_count and word_count > 0:
            errors.append(f"filler_count ({filler_count}) > word_count ({word_count})")
        
        # Check WPM is plausible
        wpm = metrics.get('speaking_rate_wpm') or metrics.get('wpm')
        if wpm is not None:
            try:
                wpm_val = float(wpm)
                if wpm_val > 350:
                    errors.append(f"WPM {wpm_val} exceeds humanly possible rate")
                elif wpm_val < 20 and word_count > 10:
                    errors.append(f"WPM {wpm_val} unrealistically slow")
            except (TypeError, ValueError):
                pass
        
        # Check score ranges
        score_keys = ['voice_confidence_score', 'vocabulary_score', 'topic_coherence_score']
        for key in score_keys:
            if key in metrics:
                value = metrics.get(key)
                if value is not None:
                    try:
                        val = float(value)
                        if val < 0 or val > 100:
                            errors.append(f"{key}: {val} is outside valid range [0, 100]")
                    except (TypeError, ValueError):
                        errors.append(f"{key}: Cannot convert to number")
        
        return errors
    
    @staticmethod
    def clamp_score(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
        """
        Clamp a score to valid range
        
        Args:
            value: The value to clamp
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clamped value
        """
        if value is None:
            return min_val
        try:
            val = float(value)
            return max(min_val, min(max_val, val))
        except (TypeError, ValueError):
            return min_val
    
    @staticmethod
    def safe_percentage(numerator: float, denominator: float, 
                        default: float = 0.0) -> float:
        """
        Safely calculate a percentage
        
        Args:
            numerator: The numerator
            denominator: The denominator
            default: Default value if calculation fails
            
        Returns:
            Calculated percentage or default
        """
        if denominator is None or denominator <= 0:
            return default
        if numerator is None or numerator < 0:
            return default
        
        try:
            result = (float(numerator) / float(denominator)) * 100
            return round(max(0.0, min(100.0, result)), 2)
        except (TypeError, ValueError, ZeroDivisionError):
            return default


# Create singleton instance
metric_validator = MetricValidator()


def validate_analysis_results(results: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validate complete analysis results and return corrected values
    
    Args:
        results: Complete analysis results dictionary
        
    Returns:
        Tuple of (all_valid, corrected_results, error_messages)
    """
    errors = []
    corrected = dict(results)
    
    # Validate and correct filler analysis
    filler_analysis = results.get('filler_analysis', {})
    if filler_analysis:
        filler_count = filler_analysis.get('filler_count', 0)
        total_words = filler_analysis.get('total_words', 0)
        reported_pct = filler_analysis.get('filler_percentage', 0)
        
        is_valid, correct_pct, error = metric_validator.validate_filler_calculation(
            filler_count, total_words, reported_pct
        )
        if not is_valid and error:
            errors.append(error)
            corrected.setdefault('filler_analysis', {})['filler_percentage'] = correct_pct
    
    # Validate speaking metrics
    speaking_metrics = results.get('speaking_metrics', {})
    if speaking_metrics:
        word_count = speaking_metrics.get('word_count', 0)
        duration = speaking_metrics.get('total_duration', 0)
        reported_wpm = speaking_metrics.get('speaking_rate_wpm', 0)
        
        if duration > 0:
            is_valid, correct_wpm, error = metric_validator.validate_wpm_calculation(
                word_count, duration, reported_wpm
            )
            if not is_valid and error:
                errors.append(error)
                corrected.setdefault('speaking_metrics', {})['speaking_rate_wpm'] = correct_wpm
    
    # Validate pause analysis
    pauses = results.get('pauses', [])
    duration = results.get('speaking_metrics', {}).get('total_duration', 0)
    if pauses and duration > 0:
        is_valid, error = metric_validator.validate_pause_analysis(pauses, duration)
        if not is_valid and error:
            errors.append(error)
    
    # Detect impossible values
    impossible_errors = metric_validator.detect_impossible_values(results)
    errors.extend(impossible_errors)
    
    # Clamp all scores to valid ranges
    score_keys = ['voice_confidence_score', 'vocabulary_score', 'topic_coherence_score', 
                  'facial_confidence_score', 'overall_score']
    for key in score_keys:
        if key in corrected:
            corrected[key] = metric_validator.clamp_score(corrected[key])
    
    all_valid = len(errors) == 0
    return (all_valid, corrected, errors)
