"""
Quick test script to verify pause detection is working via API
Tests the /api/report/{session_id} endpoint for pause_cadence data
"""
import requests
import json
import sys

API_BASE_URL = "http://localhost:8000"

def test_pause_detection():
    """Test if pause_cadence is present and has valid data"""
    
    print("=" * 60)
    print("PAUSE DETECTION TEST")
    print("=" * 60)
    
    # Step 1: Get list of sessions
    print("\n[1] Fetching available sessions...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/sessions", timeout=5)
        if response.status_code != 200:
            print(f"❌ Failed to get sessions: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
        
        sessions = response.json()
        if not sessions or len(sessions) == 0:
            print("⚠️  No sessions found. Upload a video first to test pause detection.")
            return False
        
        # Get the most recent session
        latest_session = sessions[0] if isinstance(sessions, list) else sessions.get('sessions', [])[0] if isinstance(sessions, dict) else None
        session_id = latest_session.get('session_id') if isinstance(latest_session, dict) else latest_session
        
        if not session_id:
            print("❌ Could not extract session_id from response")
            return False
        
        print(f"✓ Found session: {session_id}")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed. Is the backend running at {API_BASE_URL}?")
        print("   Start it with: python run_server.py")
        return False
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
        return False
    
    # Step 2: Get report for this session
    print(f"\n[2] Fetching report for session {session_id}...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/report/{session_id}", timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get report: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
        
        report = response.json()
        print(f"✓ Report fetched successfully")
        
    except Exception as e:
        print(f"❌ Error getting report: {e}")
        return False
    
    # Step 3: Check for pause_cadence in audio_analytics
    print(f"\n[3] Checking pause_cadence data...")
    
    audio_analytics = report.get('audio_analytics', {})
    if not audio_analytics:
        print("❌ audio_analytics not found in report")
        print(f"   Available keys: {list(report.keys())[:10]}")
        return False
    
    pause_cadence = audio_analytics.get('pause_cadence', {})
    if not pause_cadence:
        print("❌ pause_cadence not found in audio_analytics")
        print(f"   Available keys in audio_analytics: {list(audio_analytics.keys())}")
        return False
    
    print("✓ pause_cadence found!")
    
    # Step 4: Validate pause_cadence structure
    print(f"\n[4] Validating pause_cadence structure...")
    
    counts = pause_cadence.get('counts', {})
    durations = pause_cadence.get('durations', {})
    avg_duration = pause_cadence.get('average_duration')
    total_pause_time = pause_cadence.get('total_pause_time')
    
    print(f"\n   Counts: {json.dumps(counts, indent=6)}")
    print(f"   Durations: {json.dumps(durations, indent=6)}")
    print(f"   Average Duration: {avg_duration}")
    print(f"   Total Pause Time: {total_pause_time}")
    
    # Validate structure
    required_keys = ['short', 'medium', 'long']
    missing_counts = [k for k in required_keys if k not in counts]
    missing_durations = [k for k in required_keys if k not in durations]
    
    if missing_counts or missing_durations:
        print(f"\n❌ Invalid structure!")
        if missing_counts:
            print(f"   Missing in counts: {missing_counts}")
        if missing_durations:
            print(f"   Missing in durations: {missing_durations}")
        return False
    
    print("\n✓ Structure is valid (all buckets present)")
    
    # Step 5: Check if pauses are actually detected
    print(f"\n[5] Checking if pauses were detected...")
    
    total_pauses = sum(counts.values())
    total_duration = sum(durations.values())
    
    print(f"   Total pauses: {total_pauses}")
    print(f"   Total pause duration: {total_duration}s")
    
    # Also check pause_summary for comparison
    pause_summary = report.get('pause_summary', {})
    summary_total = pause_summary.get('total_pauses', 0)
    
    print(f"   Pause summary total: {summary_total}")
    
    if total_pauses == 0 and summary_total == 0:
        print("\n⚠️  WARNING: No pauses detected!")
        print("   This could mean:")
        print("   - Video has no pauses (unlikely)")
        print("   - Pause detection is not working")
        print("   - Word timestamps are not enabled")
        
        # Check if words_with_timing exists
        audio_results = report.get('audio_results', {})
        words_with_timing = audio_results.get('words_with_timing', [])
        if not words_with_timing:
            words_with_timing = report.get('words_with_timing', [])
        
        if not words_with_timing:
            print("\n   ❌ words_with_timing not found - this is likely the issue!")
            print("   Check Whisper transcription has word_timestamps=True")
            return False
        else:
            print(f"\n   ✓ words_with_timing found ({len(words_with_timing)} words)")
            print("   Pause detection may need word timing gaps extraction")
            return False
    else:
        print("\n✓ Pauses detected!")
        print(f"\n   Breakdown:")
        print(f"   - SHORT (<1s): {counts.get('short', 0)} pauses, {durations.get('short', 0)}s total")
        print(f"   - MEDIUM (1-2.5s): {counts.get('medium', 0)} pauses, {durations.get('medium', 0)}s total")
        print(f"   - LONG (>=2.5s): {counts.get('long', 0)} pauses, {durations.get('long', 0)}s total")
        return True

if __name__ == "__main__":
    print("\nTesting pause detection via API...")
    print(f"Backend URL: {API_BASE_URL}\n")
    
    success = test_pause_detection()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED: Pause detection is working!")
    else:
        print("❌ TEST FAILED: Pause detection has issues")
    print("=" * 60 + "\n")
    
    sys.exit(0 if success else 1)
