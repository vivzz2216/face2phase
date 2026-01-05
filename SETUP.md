# Face2Phase - Setup Guide

Complete setup guide for new users to get Face2Phase up and running.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11.9** (Required - other versions may have compatibility issues)
  - Download from: https://www.python.org/downloads/release/python-3119/
  - ⚠️ **Important:** Python 3.12+ and Python 3.10 or earlier are not recommended due to dependency compatibility issues
  
- **Node.js 16+** (for frontend)
  - Download from: https://nodejs.org/
  - Recommended: Node.js 18.x LTS

- **Git** (optional, if cloning from repository)

- **At least 4GB RAM** (8GB recommended for better performance)

## Step-by-Step Installation

### 1. Clone or Download the Project

If you have the repository:
```bash
git clone <repository-url>
cd Face2Phase
```

Or extract the project folder if you downloaded it as a zip file.

### 2. Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

After activation, you should see `(venv)` in your terminal prompt.

⚠️ **Important:** Make sure you're using Python 3.11.9. Check with:
```bash
python --version
# Should show: Python 3.11.9
```

If it shows a different version, you may need to:
- Use `python3.11` instead of `python`
- Or set up Python 3.11.9 as your default Python installation

### 3. Install Python Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install Whisper for audio transcription
pip install openai-whisper

# Install tf-keras (required for DeepFace/TensorFlow)
pip install tf-keras
```

⏱️ **Note:** This installation may take 10-20 minutes depending on your internet speed, as it downloads large ML libraries like PyTorch, TensorFlow, and spaCy.

### 4. Download spaCy Language Model

The application uses spaCy for text analysis. Download the English model:

```bash
python -m spacy download en_core_web_md
```

This will download approximately 50MB of language data.

### 5. Download NLTK Data

The application uses NLTK for natural language processing. Download required data:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

### 6. Set Up Environment Variables

Create a `.env` file in the project root directory:

**Windows (PowerShell):**
```powershell
New-Item -Path .env -ItemType File
```

**Linux/Mac:**
```bash
touch .env
```

Add the following content to `.env`:

```env
# OpenAI API Key (Optional - for enhanced analysis features)
OPENAI_API_KEY=your_openai_api_key_here

# JWT Secret Key (Required for authentication)
# Generate a secure key with: python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET_KEY=your_jwt_secret_key_here

# Optional: Custom port (default is 8000)
PORT=8000
```

**Generate JWT Secret Key:**
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Copy the generated key and paste it into `.env` as `JWT_SECRET_KEY`.

### 7. Install Frontend Dependencies

Open a new terminal window (keep the backend terminal with venv activated), navigate to the project directory, and run:

```bash
npm install
```

This will install all React/Node.js dependencies for the frontend.

⏱️ **Note:** This may take a few minutes.

## Running the Application

### Start the Backend Server

Make sure your virtual environment is activated, then:

**Windows:**
```bash
venv\Scripts\activate
python run_server.py
```

**Linux/Mac:**
```bash
source venv/bin/activate
python run_server.py
```

You should see output like:
```
Starting Uvicorn server on http://127.0.0.1:8000...
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

✅ The backend API is now running at `http://localhost:8000`

### Start the Frontend Development Server

Open a **new terminal window**, navigate to the project directory, and run:

```bash
npm run dev
```

You should see output like:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

✅ The frontend is now running at `http://localhost:5173`

### Access the Application

Open your web browser and navigate to:
- **Frontend UI:** http://localhost:5173
- **Backend API Docs:** http://localhost:8000/docs
- **API Health Check:** http://localhost:8000/health

## Troubleshooting

### Common Issues

#### 1. Port Already in Use (Error: WinError 10013)

If you get a port permission error:
- Stop any existing Python processes: `taskkill /F /IM python.exe` (Windows)
- Or change the port in `run_server.py` or `.env` file

#### 2. Module Not Found Errors

If you get import errors:
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11.9)

#### 3. spaCy Model Not Found

If you get spaCy model errors:
```bash
python -m spacy download en_core_web_md
```

#### 4. TensorFlow/DeepFace Errors

If you get TensorFlow-related errors:
```bash
pip install tf-keras
pip install --upgrade tensorflow
```

#### 5. Whisper Not Found

If audio transcription is disabled:
```bash
pip install openai-whisper
```

#### 6. NLTK Data Errors

If you get NLTK data errors:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

#### 7. Frontend Build Errors

If npm install fails:
- Clear cache: `npm cache clean --force`
- Delete `node_modules` and `package-lock.json`
- Run `npm install` again

### Verification Checklist

Before reporting issues, verify:

- [ ] Python version is 3.11.9
- [ ] Virtual environment is created and activated
- [ ] All dependencies installed (`pip list` shows required packages)
- [ ] spaCy model downloaded (`python -m spacy info en_core_web_md`)
- [ ] NLTK data downloaded
- [ ] `.env` file exists with JWT_SECRET_KEY
- [ ] Backend server starts without errors
- [ ] Frontend server starts without errors
- [ ] Can access http://localhost:5173
- [ ] Can access http://localhost:8000/docs

## Project Structure

```
Face2Phase/
├── backend/              # Python backend (FastAPI)
│   ├── app.py           # Main application file
│   ├── analysis/        # Analysis modules (audio, face, text, speech)
│   ├── services/        # Business logic services
│   ├── exporters/       # Report generation
│   ├── db/              # Database models
│   └── utils/           # Utility functions
├── src/                 # React frontend
│   ├── components/      # React components
│   ├── context/         # React context (auth, etc.)
│   └── lib/             # Frontend utilities
├── storage/             # User uploads and generated files
│   ├── uploads/         # Uploaded audio/video files
│   ├── reports/         # Generated analysis reports
│   └── exports/         # Exported PDF reports
├── requirements.txt     # Python dependencies
├── package.json         # Node.js dependencies
├── run_server.py        # Server startup script
└── .env                 # Environment variables (create this)
```

## Additional Notes

### Performance Optimization

- First run may be slower as ML models are downloaded and cached
- Models are cached in `models_cache/` directory
- Subsequent runs will be faster

### Development Mode

- Backend runs with auto-reload (code changes restart server)
- Frontend runs with hot-reload (code changes refresh browser)
- Use `CTRL+C` to stop servers

### Production Deployment

For production deployment:
- Use `gunicorn` or similar WSGI server for backend
- Build frontend with `npm run build`
- Configure environment variables securely
- Use a production-grade database (PostgreSQL recommended)
- Set up proper logging and monitoring

## Support

If you encounter issues not covered in this guide:
1. Check the error messages carefully
2. Verify all prerequisites are met
3. Check that Python version is exactly 3.11.9
4. Review the troubleshooting section
5. Check application logs in `face2phrase.log`

## Quick Start Summary

For experienced users, here's the quick version:

```bash
# 1. Create venv
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt
pip install openai-whisper tf-keras
python -m spacy download en_core_web_md
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# 3. Create .env file with JWT_SECRET_KEY

# 4. Install frontend
npm install

# 5. Run backend (terminal 1)
python run_server.py

# 6. Run frontend (terminal 2)
npm run dev
```

---

**Happy coding! 🚀**
