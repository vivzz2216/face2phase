# FIX PREVIOUS ANALYSES TABLE - SHOWING DASHES

## Problem
Table shows "—" for all values instead of actual data:
- Total Time: —
- Pacing: —  
- Filler %: —
- Total Words: —

## Root Cause
The frontend is looking for fields that don't exist in the backend response.

**Current code expects:**
- `duration_seconds` or `total_time_label`
- `pacing_value`
- `filler_percentage`
- `total_words`

**Backend likely returns:**
- `duration` (not `duration_seconds`)
- `speaking_rate_wpm` (not `pacing_value`)
- `filler_word_count` and `filler_ratio` (not `filler_percentage`)
- `word_count` (not `total_words`)

## Solution 1: Simplify Columns (As Requested)

Remove unnecessary columns and keep only:
1. **Title** - File name
2. **Created** - Date/time
3. **Score** - Overall score
4. **Type** - Video/Audio
5. **Duration** - How long the recording was
6. **Voice** - Voice confidence score
7. **Facial** - Facial confidence score

### Changes Needed in Dashboard.jsx:

**Lines 1510-1538:** Update table headers
```jsx
<div className="history-table-header" role="row">
  <div className="history-col title" role="columnheader">
    <span>Title</span>
  </div>
  <div className="history-col created" role="columnheader">
    <span>Created</span>
  </div>
  <div className="history-col score" role="columnheader">
    <span>Overall</span>
  </div>
  <div className="history-col type" role="columnheader">
    <span>Type</span>
  </div>
  <div className="history-col duration" role="columnheader">
    <span>Duration</span>
  </div>
  <div className="history-col voice" role="columnheader">
    <span>Voice</span>
  </div>
  <div className="history-col facial" role="columnheader">
    <span>Facial</span>
  </div>
  <div className="history-col actions" role="columnheader" aria-label="Actions" />
</div>
```

**Lines 1602-1620:** Update table cells
```jsx
<div className="history-cell duration" role="cell">
  <span>{formatDuration(analysis.duration)}</span>
</div>

<div className="history-cell voice" role="cell">
  <span style={{ color: getScoreColor(analysis.voice_confidence) }}>
    {analysis.voice_confidence ?? '—'}
  </span>
</div>

<div className="history-cell facial" role="cell">
  <span style={{ color: getScoreColor(analysis.facial_confidence) }}>
    {analysis.facial_confidence ?? '—'}
  </span>
</div>
```

## Solution 2: Fix Backend Response

If backend doesn't return these fields, check:

**File:** `backend/exporters/report_generator.py`

Look for where session data is saved. Ensure it includes:
```python
{
    "session_id": session_id,
    "file_name": filename,
    "file_type": "video" or "audio",
    "created_at": timestamp,
    "overall_score": overall_score,
    "voice_confidence": voice_confidence,
    "facial_confidence": facial_confidence,
    "duration": duration_seconds,  # NOT duration_seconds
    "thumbnail_url": thumbnail_path
}
```

## Quick Fix (Frontend Only)

**File:** `src/components/Dashboard.jsx`

**Line 1602-1604:** Fix duration
```jsx
<div className="history-cell duration" role="cell">
  <span>
    {formatDuration(
      analysis.duration || 
      analysis.duration_seconds || 
      analysis.speaking_metrics?.total_duration
    )}
  </span>
</div>
```

**Remove lines 1606-1620** (Pacing, Filler %, Total Words columns)

**Add after line 1600:**
```jsx
<div className="history-cell voice" role="cell">
  <span style={{ color: getScoreColor(analysis.voice_confidence) }}>
    {analysis.voice_confidence ?? '—'}
  </span>
</div>

<div className="history-cell facial" role="cell">
  <span style={{ color: getScoreColor(analysis.facial_confidence) }}>
    {analysis.facial_confidence ?? '—'}
  </span>
</div>
```

## Expected Result

Table should show:
```
Title                    | Created      | Overall | Type  | Duration | Voice | Facial
WhatsApp Video...        | Jan 16 1:49  | 26      | Video | 1:58     | 28    | 100
```

Instead of:
```
Title                    | Created      | Overall | Type  | Duration | Pacing | Filler % | Total Words
WhatsApp Video...        | Jan 16 1:49  | 26      | Video | —        | —      | —        | —
```

---

**Apply these changes to Dashboard.jsx and restart the frontend.**
