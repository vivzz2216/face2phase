# Face2Phase - Installation Guide

## ✅ Complete Single-Command Installation

This project now supports **complete installation with a single command**. All dependencies are included in `requirements.txt`.

---

## 🚀 Quick Installation (3 Steps)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

**That's it!** All Python packages will be installed automatically, including:
- TensorFlow & tf-keras
- DeepFace & MTCNN
- Matplotlib & Seaborn
- FastAPI & Uvicorn
- And 30+ other dependencies

### Step 2: Install Node Dependencies
```bash
npm install
```

### Step 3: Verify Installation (Optional)
```bash
python test_imports.py
```

**Expected output:**
```
✅ SUCCESS: All critical dependencies installed!
```

---

## 🎯 Running the Application

### Start Backend (Terminal 1)
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### Start Frontend (Terminal 2)
```bash
npm run dev
```

### Access the Application
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:8000

---

## 📋 What's Included in requirements.txt

### Core Framework (5 packages)
- fastapi==0.110.0
- uvicorn[standard]==0.27.1
- pydantic==2.6.4
- python-multipart==0.0.9

### Numerical Computing (3 packages)
- numpy>=1.20,<2 (compatible with matplotlib)
- scipy==1.11.4
- numba==0.56.4

### Audio Processing (3 packages)
- librosa==0.9.2
- soundfile==0.12.1
- openai-whisper (for speech-to-text transcription)

### Computer Vision (4 packages)
- opencv-python==4.8.1.78
- mtcnn==0.1.1
- deepface==0.0.97
- Pillow==10.2.0

### Deep Learning (2 packages)
- tensorflow>=2.20.0,<2.21
- tf-keras>=2.20.0

### NLP (7 packages)
- spacy==3.7.4
- nltk==3.8.1
- transformers==4.37.2
- huggingface-hub==0.20.3
- tokenizers==0.15.2
- sentencepiece==0.1.99
- safetensors==0.4.2

### Data Processing (2 packages)
- pandas==2.0.3
- scikit-learn==1.3.2

### Video Processing (2 packages)
- moviepy==2.1.1
- imageio-ffmpeg==0.6.0

### Utilities (3 packages)
- python-dotenv==1.0.1
- aiofiles==23.2.1
- jinja2==3.1.3

### Authentication & Security (3 packages)
- python-jose[cryptography]==3.3.0
- passlib[bcrypt]==1.7.4
- email-validator==2.1.0.post1

### Database (2 packages)
- SQLAlchemy==2.0.27
- alembic==1.13.1

### AI Integration (1 package)
- openai==1.12.0

### Reporting (4 packages)
- reportlab==4.1.0
- python-docx==1.1.0
- matplotlib==3.7.4
- seaborn==0.13.2

### Testing (2 packages)
- pytest==7.4.4
- pytest-asyncio==0.21.1

**Total: 50+ packages installed automatically**

---

## ⚠️ Optional Packages

### webrtcvad (Voice Activity Detection)
**Note:** Requires Microsoft Visual C++ Build Tools on Windows

```bash
pip install webrtcvad
```

This is **optional** - the system works fine without it.

### spaCy Language Model (Better Text Analysis)
```bash
python -m spacy download en_core_web_md
```

This is **optional** - basic text analysis works without it.

---

## 🔧 System Requirements

- **Python:** 3.10.11 or higher
- **Node.js:** Latest LTS version
- **OS:** Windows, macOS, or Linux
- **GPU:** Optional (CUDA-enabled GPU for faster processing)

---

## ✅ Verification

After installation, verify everything works:

```bash
python test_imports.py
```

Expected output:
```
Testing critical dependencies...

✅ FastAPI
✅ Uvicorn
✅ Pydantic
✅ NumPy
✅ TensorFlow
✅ tf-keras
✅ DeepFace
✅ Matplotlib
... (30+ more packages)

✅ SUCCESS: All critical dependencies installed!
```

---

## 🐛 Troubleshooting

### Issue: Port already in use
**Solution:** Use a different port or stop the conflicting process
```bash
# Use different port
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8001
```

### Issue: NumPy version conflict
**Solution:** Already handled in requirements.txt with `numpy>=1.20,<2`

### Issue: TensorFlow import error
**Solution:** tf-keras is now included in requirements.txt

---

## 📚 Documentation

- **Full Error Report:** See `PROJECT_RUN_REPORT.md`
- **Project README:** See `README.md`
- **API Documentation:** http://localhost:8000/docs (when backend is running)

---

## 🎉 Success!

If you can access http://localhost:5173 and see the Face2Phase interface, you're all set!

The application is ready for:
- Video analysis
- Audio analysis
- Speech evaluation
- Facial expression analysis
- Comprehensive feedback generation
