"""
Direct test of pause detection logic - tests the actual pause extraction functions
without needing the full API/server running
"""
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.utils.report_utils import compute_pause_cadence

def test_compute_pause_cadence():
    """Test pause cadence computation with sample data"""
    
    print("=" * 60)
    print("DIRECT PAUSE CADENCE TEST")
    print("=" * 60)
    print()
    
    # Test 1: Empty list
    print("[Test 1] Empty pauses list...")
    result = compute_pause_cadence([])
    print(f"   Result: {result}")
    assert result['counts']['short'] == 0, "Should be 0"
    assert result['counts']['medium'] == 0, "Should be 0"
    assert result['counts']['long'] == 0, "Should be 0"
    print("   [OK] Empty list handled correctly")
    print()
    
    # Test 2: None input
    print("[Test 2] None input...")
    result = compute_pause_cadence(None)
    print(f"   Result: {result}")
    assert result['counts']['short'] == 0, "Should be 0"
    print("   [OK] None handled correctly")
    print()
    
    # Test 3: Valid pauses
    print("[Test 3] Valid pauses...")
    test_pauses = [
        {"start": 1.0, "end": 1.5, "duration": 0.5, "type": "SHORT_PAUSE"},  # Short
        {"start": 3.0, "end": 4.2, "duration": 1.2, "type": "MEDIUM_PAUSE"},  # Medium
        {"start": 6.0, "end": 9.0, "duration": 3.0, "type": "LONG_PAUSE"},    # Long
        {"start": 10.0, "end": 10.8, "duration": 0.8, "type": "SHORT_PAUSE"}, # Short
        {"start": 12.0, "end": 13.5, "duration": 1.5, "type": "MEDIUM_PAUSE"}, # Medium
    ]
    result = compute_pause_cadence(test_pauses)
    print(f"   Result: {result}")
    
    assert result['counts']['short'] == 2, f"Expected 2 short, got {result['counts']['short']}"
    assert result['counts']['medium'] == 2, f"Expected 2 medium, got {result['counts']['medium']}"
    assert result['counts']['long'] == 1, f"Expected 1 long, got {result['counts']['long']}"
    assert result['average_duration'] > 0, "Average should be > 0"
    
    print(f"   [OK] Correctly classified:")
    print(f"     - SHORT: {result['counts']['short']} ({result['durations']['short']}s)")
    print(f"     - MEDIUM: {result['counts']['medium']} ({result['durations']['medium']}s)")
    print(f"     - LONG: {result['counts']['long']} ({result['durations']['long']}s)")
    print(f"     - Avg: {result['average_duration']}s")
    print()
    
    # Test 4: Invalid pause data
    print("[Test 4] Invalid pause data (should skip invalid)...")
    invalid_pauses = [
        {"start": 1.0, "end": 2.0, "duration": 1.0},  # Valid
        {"start": 3.0, "duration": "invalid"},        # Invalid duration
        {"duration": -1.0},                           # Negative duration
        {"duration": None},                           # None duration
        {"start": 5.0, "end": 6.5, "duration": 1.5},  # Valid
    ]
    result = compute_pause_cadence(invalid_pauses)
    print(f"   Result: {result}")
    # Should process 2 valid pauses
    assert result['counts']['short'] + result['counts']['medium'] + result['counts']['long'] == 2
    print("   [OK] Invalid pauses filtered correctly")
    print()
    
    print("=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        test_compute_pause_cadence()
    except Exception as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
