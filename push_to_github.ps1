# PowerShell script to push all changes to GitHub
# Run this after installing Git

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Push Changes to GitHub" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is available
try {
    $gitVersion = git --version
    Write-Host "[OK] Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Then restart PowerShell and run this script again." -ForegroundColor Yellow
    exit 1
}

# Navigate to project directory
$projectPath = "C:\Users\ACER\Desktop\face2phase-master"
Set-Location $projectPath
Write-Host "[OK] Changed to project directory" -ForegroundColor Green

# Check if .git exists, if not initialize
if (-not (Test-Path ".git")) {
    Write-Host "[INFO] Initializing git repository..." -ForegroundColor Yellow
    git init
}

# Check remote
$remoteExists = git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[INFO] Adding remote repository..." -ForegroundColor Yellow
    git remote add origin https://github.com/vivzz2216/face2phase.git
} else {
    Write-Host "[INFO] Remote already exists: $remoteExists" -ForegroundColor Yellow
    Write-Host "[INFO] Updating remote URL..." -ForegroundColor Yellow
    git remote set-url origin https://github.com/vivzz2216/face2phase.git
}

# Show status
Write-Host ""
Write-Host "[INFO] Current git status:" -ForegroundColor Cyan
git status

# Add all changes
Write-Host ""
Write-Host "[INFO] Adding all changes..." -ForegroundColor Yellow
git add .

# Show what will be committed
Write-Host ""
Write-Host "[INFO] Changes to be committed:" -ForegroundColor Cyan
git status --short

# Commit
Write-Host ""
$commitMessage = "Update: Fix pause detection with word timestamps, remove hardcoded data, enhance analytics sections"
Write-Host "[INFO] Committing changes with message: $commitMessage" -ForegroundColor Yellow
git commit -m $commitMessage

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Commit failed or no changes to commit" -ForegroundColor Yellow
}

# Determine branch
$currentBranch = git branch --show-current 2>$null
if (-not $currentBranch) {
    $currentBranch = "master"
    Write-Host "[INFO] Creating master branch..." -ForegroundColor Yellow
    git checkout -b master
}

Write-Host ""
Write-Host "[INFO] Current branch: $currentBranch" -ForegroundColor Cyan

# Push to GitHub
Write-Host ""
Write-Host "[INFO] Pushing to GitHub..." -ForegroundColor Yellow
Write-Host "Note: You will be prompted for GitHub credentials." -ForegroundColor Yellow
Write-Host "Use a Personal Access Token as password (not your GitHub password)." -ForegroundColor Yellow
Write-Host ""

# Try pushing
try {
    git push -u origin $currentBranch
    Write-Host ""
    Write-Host "[SUCCESS] Changes pushed to GitHub successfully!" -ForegroundColor Green
    Write-Host "Repository: https://github.com/vivzz2216/face2phase" -ForegroundColor Cyan
} catch {
    Write-Host ""
    Write-Host "[ERROR] Push failed. Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible issues:" -ForegroundColor Yellow
    Write-Host "1. Authentication failed - make sure you're using a Personal Access Token" -ForegroundColor Yellow
    Write-Host "2. Branch name mismatch - the remote might use 'main' instead of 'master'" -ForegroundColor Yellow
    Write-Host "3. Network issues - check your internet connection" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To create a Personal Access Token:" -ForegroundColor Cyan
    Write-Host "1. Go to: https://github.com/settings/tokens" -ForegroundColor Cyan
    Write-Host "2. Generate new token (classic)" -ForegroundColor Cyan
    Write-Host "3. Select 'repo' scope" -ForegroundColor Cyan
    Write-Host "4. Copy the token and use it as your password" -ForegroundColor Cyan
}
