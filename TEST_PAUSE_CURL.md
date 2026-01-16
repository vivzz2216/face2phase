# Quick Pause Detection Test via CURL

## Test Commands (Run in PowerShell)

### Step 1: Get a session ID
```powershell
curl.exe -s http://localhost:8000/api/sessions
```

### Step 2: Get report and extract pause_cadence (replace SESSION_ID)
```powershell
$sessionId = "YOUR_SESSION_ID_HERE"
curl.exe -s "http://localhost:8000/api/report/$sessionId" | ConvertFrom-Json | Select-Object -ExpandProperty audio_analytics | Select-Object -ExpandProperty pause_cadence
```

### Step 3: Check if pauses are detected
```powershell
$report = curl.exe -s "http://localhost:8000/api/report/$sessionId" | ConvertFrom-Json
$pauseCadence = $report.audio_analytics.pause_cadence
Write-Host "Counts: $($pauseCadence.counts | ConvertTo-Json)"
Write-Host "Total pauses: $($pauseCadence.counts.short + $pauseCadence.counts.medium + $pauseCadence.counts.long)"
```

### Step 4: Check words_with_timing (to verify word timestamps)
```powershell
$report = curl.exe -s "http://localhost:8000/api/report/$sessionId" | ConvertFrom-Json
$words = $report.audio_results.words_with_timing
if ($words) {
    Write-Host "✓ words_with_timing found: $($words.Count) words"
} else {
    Write-Host "❌ words_with_timing NOT FOUND"
}
```

## One-Liner Test (Quick Check)

```powershell
$id = (curl.exe -s http://localhost:8000/api/sessions | ConvertFrom-Json)[0].session_id; $r = curl.exe -s "http://localhost:8000/api/report/$id" | ConvertFrom-Json; Write-Host "Pauses: $($r.audio_analytics.pause_cadence.counts | ConvertTo-Json)"
```

---

## Expected Output

If working correctly:
```json
{
  "short": 5,
  "medium": 3,
  "long": 1
}
```

If not working:
```json
{
  "short": 0,
  "medium": 0,
  "long": 0
}
```
