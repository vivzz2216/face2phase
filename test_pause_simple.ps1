# Simple pause detection test
$sessionId = (curl.exe -s http://localhost:8000/api/sessions | ConvertFrom-Json)[0].session_id
$report = curl.exe -s "http://localhost:8000/api/report/$sessionId" | ConvertFrom-Json
$pauseCadence = $report.audio_analytics.pause_cadence
$total = $pauseCadence.counts.short + $pauseCadence.counts.medium + $pauseCadence.counts.long
Write-Host "Pause Cadence: SHORT=$($pauseCadence.counts.short) MEDIUM=$($pauseCadence.counts.medium) LONG=$($pauseCadence.counts.long) TOTAL=$total"
