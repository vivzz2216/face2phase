# Instructions to Push Changes to GitHub

Git is not currently installed on your system. Follow these steps to push your changes:

## Step 1: Install Git

1. Download Git for Windows from: https://git-scm.com/download/win
2. Run the installer with default options
3. Restart your terminal/PowerShell after installation

## Step 2: Configure Git (First Time Only)

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 3: Navigate to Project Directory

```powershell
cd C:\Users\ACER\Desktop\face2phase-master
```

## Step 4: Initialize Git Repository (If Not Already Initialized)

```powershell
git init
```

## Step 5: Add Remote Repository

```powershell
git remote add origin https://github.com/vivzz2216/face2phase.git
```

Or if remote already exists, update it:
```powershell
git remote set-url origin https://github.com/vivzz2216/face2phase.git
```

## Step 6: Add All Changes

```powershell
git add .
```

## Step 7: Commit Changes

```powershell
git commit -m "Update: Remove dummy data, fix pause detection, and improve analytics"
```

Or a more detailed commit message:
```powershell
git commit -m "Update: Fix pause detection with word timestamps, remove hardcoded data, enhance analytics sections"
```

## Step 8: Push to GitHub

### If pushing for the first time:
```powershell
git push -u origin master
```

### If master branch exists on remote:
```powershell
git push origin master
```

### If the remote uses 'main' branch:
```powershell
git push -u origin main
```

## Step 9: Authentication

When prompted:
- **Username**: Your GitHub username (vivzz2216)
- **Password**: Use a **Personal Access Token** (NOT your GitHub password)

### Create Personal Access Token:
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "face2phase-push"
4. Select scopes: Check "repo" (full control of private repositories)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. Use this token as your password when pushing

## Alternative: Using GitHub Desktop

If you prefer a GUI:
1. Download GitHub Desktop: https://desktop.github.com/
2. Sign in with your GitHub account
3. File → Add Local Repository
4. Select: `C:\Users\ACER\Desktop\face2phase-master`
5. Commit changes with a message
6. Click "Push origin"

## Troubleshooting

### If you get "remote origin already exists":
```powershell
git remote remove origin
git remote add origin https://github.com/vivzz2216/face2phase.git
```

### If you get "failed to push some refs":
You may need to pull first:
```powershell
git pull origin master --allow-unrelated-histories
git push origin master
```

### Check current status:
```powershell
git status
git remote -v
git branch
```

---

## Quick Command Summary

Once Git is installed, run these commands in order:

```powershell
cd C:\Users\ACER\Desktop\face2phase-master
git init
git remote add origin https://github.com/vivzz2216/face2phase.git
git add .
git commit -m "Update: Fix pause detection, remove dummy data, enhance analytics"
git push -u origin master
```
