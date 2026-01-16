# ✅ PREVIOUS ANALYSES TABLE - FIXED

## Changes Made

### 1. Frontend - Dashboard.jsx

**Simplified Table Columns:**
- ❌ Removed: Pacing, Filler %, Total Words
- ✅ Added: Voice Confidence, Facial Confidence
- ✅ Kept: Title, Created, Overall Score, Type, Duration

**Table Headers (Lines 1510-1538):**
```jsx
- Overall (instead of "Score")
- Duration (instead of "Total Time")
- Voice (new)
- Facial (new)
```

**Table Cells (Lines 1602-1620):**
```jsx
// Duration - tries multiple field names
{formatDuration(
  analysis.duration || 
  analysis.duration_seconds || 
  analysis.speaking_metrics?.total_duration ||
  analysis.total_duration
)}

// Voice Confidence - with color coding
<span style={{ color: getScoreColor(analysis.voice_confidence) }}>
  {analysis.voice_confidence != null ? Math.round(Number(analysis.voice_confidence)) : '—'}
</span>

// Facial Confidence - with color coding
<span style={{ color: getScoreColor(analysis.facial_confidence) }}>
  {analysis.facial_confidence != null ? Math.round(Number(analysis.facial_confidence)) : '—'}
</span>
```

### 2. Backend - report_generator.py

**Added Duration to Metrics (Line 578):**
```python
metrics = {
    "duration": report.get("duration"),  # NEW
    "filler_word_count": report.get("filler_word_count"),
    # ... rest of metrics
}
```

### 3. Backend - database.py

**Exposed Duration in API Response (Line 262):**
```python
results.append({
    # ... other fields
    "voice_confidence": score_breakdown.get("voice_confidence"),
    "facial_confidence": score_breakdown.get("facial_confidence"),
    "vocabulary_score": score_breakdown.get("vocabulary_score"),
    "duration": metrics.get("duration"),  # NEW
    "thumbnail_url": highlights.get("thumbnail_url"),
})
```

---

## Expected Result

### Before:
```
Title                    | Created      | Score | Type  | Total Time | Pacing | Filler % | Total Words
WhatsApp Video...        | Jan 16 1:49  | 26    | Video | —          | —      | —        | —
```

### After:
```
Title                    | Created      | Overall | Type  | Duration | Voice | Facial
WhatsApp Video...        | Jan 16 1:49  | 26      | Video | 1:58     | 28    | 100
```

---

## How It Works

1. **When video is analyzed:**
   - `report_generator.build_session_summary()` extracts duration from report
   - Saves it in `metrics` JSON field in database

2. **When history is loaded:**
   - `database.get_user_sessions()` retrieves session summaries
   - Extracts `duration` from `metrics` JSON
   - Returns it as top-level field in response

3. **Frontend displays:**
   - Tries multiple field names for backward compatibility
   - Shows duration in MM:SS format
   - Shows voice/facial scores with color coding (green/yellow/red)

---

## Testing

1. **Upload a new video** (existing sessions won't have duration yet)
2. **Check Previous Analyses table:**
   - ✅ Duration should show (e.g., "1:58")
   - ✅ Voice score should show (e.g., "28")
   - ✅ Facial score should show (e.g., "100")
   - ✅ No more "—" dashes

3. **For old sessions:**
   - Duration might still show "—" (not saved in old data)
   - Voice/Facial should work if those were saved

---

## Files Modified

1. ✅ `src/components/Dashboard.jsx` - Table UI
2. ✅ `backend/exporters/report_generator.py` - Save duration
3. ✅ `backend/db/database.py` - Expose duration in API

**Server restart:** Backend changes require server restart (already running)
**Frontend:** Hot reload should pick up changes automatically
