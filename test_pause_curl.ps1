# PowerShell script to test pause detection via curl
$API_BASE = "http://localhost:8000"

Write-Host ("=" * 60)
Write-Host "PAUSE DETECTION TEST (CURL)" -ForegroundColor Cyan
Write-Host ("=" * 60)
Write-Host ""

# Step 1: Get list of sessions
Write-Host "[1] Fetching available sessions..." -ForegroundColor Yellow
try {
    $sessionsResponse = curl.exe -s "$API_BASE/api/sessions"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to connect to backend. Is it running?" -ForegroundColor Red
        Write-Host "   Start with: python run_server.py" -ForegroundColor Yellow
        exit 1
    }
    
    $sessions = $sessionsResponse | ConvertFrom-Json
    if (-not $sessions -or $sessions.Count -eq 0) {
        Write-Host "⚠️  No sessions found. Upload a video first." -ForegroundColor Yellow
        exit 1
    }
    
    # Get first session ID
    $sessionId = $sessions[0].session_id
    if (-not $sessionId) {
        $sessionId = $sessions[0]
        if (-not $sessionId -or $sessionId -isnot [string]) {
            Write-Host "❌ Could not extract session_id" -ForegroundColor Red
            Write-Host "Response: $sessionsResponse" -ForegroundColor Gray
            exit 1
        }
    }
    
    Write-Host "✓ Found session: $sessionId" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
    exit 1
}

# Step 2: Get report
Write-Host ""
Write-Host "[2] Fetching report..." -ForegroundColor Yellow
try {
    $reportResponse = curl.exe -s "$API_BASE/api/report/$sessionId"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to get report" -ForegroundColor Red
        exit 1
    }
    
    $report = $reportResponse | ConvertFrom-Json
    Write-Host "✓ Report fetched" -ForegroundColor Green
} catch {
    Write-Host "❌ Error getting report: $_" -ForegroundColor Red
    exit 1
}

# Step 3: Check pause_cadence
Write-Host ""
Write-Host "[3] Checking pause_cadence..." -ForegroundColor Yellow

if (-not $report.audio_analytics) {
    Write-Host "❌ audio_analytics not found in report" -ForegroundColor Red
    exit 1
}

if (-not $report.audio_analytics.pause_cadence) {
    Write-Host "❌ pause_cadence not found in audio_analytics" -ForegroundColor Red
    $keys = $report.audio_analytics.PSObject.Properties.Name -join ', '
    Write-Host "Available keys: $keys" -ForegroundColor Gray
    exit 1
}

$pauseCadence = $report.audio_analytics.pause_cadence
Write-Host "✓ pause_cadence found!" -ForegroundColor Green

# Step 4: Display pause data
Write-Host ""
Write-Host "[4] Pause Cadence Data:" -ForegroundColor Yellow
Write-Host ""

$counts = $pauseCadence.counts
$durations = $pauseCadence.durations
$avgDuration = $pauseCadence.average_duration
$totalPauseTime = $pauseCadence.total_pause_time

Write-Host "   Counts:" -ForegroundColor Cyan
Write-Host "     - SHORT:  $($counts.short) pauses" -ForegroundColor White
Write-Host "     - MEDIUM: $($counts.medium) pauses" -ForegroundColor White
Write-Host "     - LONG:   $($counts.long) pauses" -ForegroundColor White

Write-Host ""
Write-Host "   Durations:" -ForegroundColor Cyan
Write-Host "     - SHORT:  $($durations.short)s total" -ForegroundColor White
Write-Host "     - MEDIUM: $($durations.medium)s total" -ForegroundColor White
Write-Host "     - LONG:   $($durations.long)s total" -ForegroundColor White

Write-Host ""
Write-Host "   Average Duration: ${avgDuration}s" -ForegroundColor Cyan
Write-Host "   Total Pause Time: ${totalPauseTime}s" -ForegroundColor Cyan

# Step 5: Validate
Write-Host ""
Write-Host "[5] Validation..." -ForegroundColor Yellow

$totalPauses = $counts.short + $counts.medium + $counts.long
$totalDuration = $durations.short + $durations.medium + $durations.long

if ($totalPauses -eq 0) {
    Write-Host ""
    Write-Host "⚠️  WARNING: No pauses detected (all zeros)" -ForegroundColor Yellow
    Write-Host ""
    
    # Check pause_summary for comparison
    $pauseSummary = $report.pause_summary
    if ($pauseSummary) {
        $summaryTotal = $pauseSummary.total_pauses
        Write-Host "   pause_summary.total_pauses: $summaryTotal" -ForegroundColor Gray
    }
    
    # Check words_with_timing
    $audioResults = $report.audio_results
    $wordsWithTiming = $null
    if ($audioResults -and $audioResults.words_with_timing) {
        $wordsWithTiming = $audioResults.words_with_timing
    }
    if (-not $wordsWithTiming -and $report.words_with_timing) {
        $wordsWithTiming = $report.words_with_timing
    }
    
    if (-not $wordsWithTiming) {
        Write-Host ""
        Write-Host "   ❌ words_with_timing NOT FOUND" -ForegroundColor Red
        Write-Host "   This means Whisper word timestamps are not available" -ForegroundColor Red
        Write-Host "   Pause detection requires word-level timestamps!" -ForegroundColor Red
        exit 1
    } else {
        Write-Host ""
        Write-Host "   ✓ words_with_timing found: $($wordsWithTiming.Count) words" -ForegroundColor Green
        Write-Host "   But no pauses detected - may need fallback extraction" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "✅ SUCCESS: Pauses detected!" -ForegroundColor Green
Write-Host "   Total: $totalPauses pauses, $totalDuration seconds" -ForegroundColor Green
exit 0
