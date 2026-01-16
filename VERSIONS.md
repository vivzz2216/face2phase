# Package Versions & Dependencies

This document lists all package versions used in the Face2Phase project, with special attention to critical packages like `numba` and `numpy`.

---

## 🎯 Critical Packages: numba & numpy

### **numba**
- **Version Constraint**: `numba>=0.56.0,<0.57.0`
- **Recommended**: `numba==0.57.1` (for Python 3.11+)
- **Purpose**: JIT compiler for numerical functions (used by librosa for pitch tracking)
- **Important Notes**:
  - ⚠️ **JIT is DISABLED** in this project to avoid compatibility issues with librosa
  - Numba JIT compilation is disabled via environment variables (`NUMBA_DISABLE_JIT=1`)
  - This is a workaround for `librosa.pyin` compatibility issues with numba 0.56.x/0.57.x
  - Files with numba disabling: `backend/app.py`, `backend/core/settings.py`, `backend/analysis/audio/*.py`

### **numpy**
- **Version Constraint**: `numpy>=1.21.0,<1.25.0`
- **Purpose**: Core numerical computing library
- **Important Notes**:
  - Upper bound (`<1.25.0`) is critical for compatibility with:
    - `librosa` 0.10.x
    - `numba` 0.56.x/0.57.x
    - `tensorflow` 2.15.x
  - Lower bound (`>=1.21.0`) required for modern features used by ML libraries

---

## 🎵 Audio Processing Packages

### **librosa**
- **Version**: `librosa>=0.10.0,<0.11.0`
- **Purpose**: Audio analysis, pitch tracking, feature extraction
- **Critical Constraint**: Must stay below 0.11.0 to avoid numba compatibility issues
- **Note**: Used with numba JIT disabled to prevent `TypingError` in `librosa.pyin`

### **soundfile**
- **Version**: `soundfile>=0.12.0`
- **Purpose**: Audio file I/O (WAV, FLAC, etc.)

### **openai-whisper**
- **Version**: `openai-whisper>=20231117`
- **Purpose**: Speech-to-text transcription with word-level timestamps
- **Model**: `large-v3` (configured in settings)

### **webrtcvad-wheels**
- **Version**: `webrtcvad-wheels>=2.0.10.post2`
- **Purpose**: Voice Activity Detection (VAD) with prebuilt wheels for Windows/Python 3.13

---

## 🤖 AI/ML Packages

### **DeepFace**
- **Version**: `deepface>=0.0.79`
- **Purpose**: Facial emotion detection, age, gender analysis

### **TensorFlow/Keras**
- **Version**: `tf-keras>=2.15.0`
- **Purpose**: Required by DeepFace
- **Note**: Compatible with `numpy<1.25.0` and `tensorflow==2.15.1`

### **OpenCV**
- **Version**: `opencv-python>=4.8.0`
- **Purpose**: Image/video processing, face detection

### **spaCy**
- **Version**: `spacy>=3.7.0`
- **Model**: `en_core_web_md` (required, download separately)
- **Purpose**: NLP, entity recognition, POS tagging

### **Transformers**
- **Version**: `transformers>=4.35.0`
- **Purpose**: HuggingFace models for disfluency detection

### **huggingface-hub**
- **Version**: `huggingface-hub>=0.23.0`
- **Purpose**: Download and manage HuggingFace models

### **sentencepiece**
- **Version**: `sentencepiece>=0.1.99`
- **Purpose**: Tokenization for transformer models

### **NLTK**
- **Version**: `nltk>=3.8.0`
- **Purpose**: Natural language processing utilities

### **scikit-learn**
- **Version**: `scikit-learn>=1.3.0`
- **Purpose**: ML pattern recognition for weak word detection (Phase 3 improvement)

### **scipy**
- **Version**: `scipy>=1.11.0`
- **Purpose**: Scientific computing for acoustic analysis (Phase 3)

### **MTCNN**
- **Version**: `mtcnn>=0.1.1`
- **Purpose**: Modern face detection (Phase 1 accuracy improvement)

---

## 🧠 Deep Learning Packages

### **PyTorch**
- **Version**: `torch>=2.0.0`
- **Purpose**: Deep learning framework (used by Whisper)

### **TorchAudio**
- **Version**: `torchaudio>=2.0.0`
- **Purpose**: Audio processing for PyTorch

---

## 📊 Data Processing Packages

### **pandas**
- **Version**: `pandas>=1.5.0`
- **Purpose**: Data manipulation and analysis

### **Pillow**
- **Version**: `pillow>=9.0.0`
- **Purpose**: Image processing

---

## 🌐 Web Framework Packages

### **FastAPI**
- **Version**: `fastapi>=0.104.1`
- **Purpose**: Modern Python web framework

### **Uvicorn**
- **Version**: `uvicorn[standard]>=0.24.0`
- **Purpose**: ASGI server for FastAPI

### **python-multipart**
- **Version**: `python-multipart>=0.0.6`
- **Purpose**: File upload support for FastAPI

---

## 🎬 Video Processing Packages

### **moviepy**
- **Version**: `moviepy>=2.2.0`
- **Purpose**: Video editing and processing

### **imageio-ffmpeg**
- **Version**: `imageio-ffmpeg>=0.6.0`
- **Purpose**: FFmpeg backend for video processing

---

## 🔐 Authentication & Security

### **python-jose**
- **Version**: `python-jose[cryptography]>=3.3.0,<4.0.0`
- **Purpose**: JWT token encoding/decoding

### **passlib**
- **Version**: `passlib[bcrypt]>=1.7.4,<2.0.0`
- **Purpose**: Password hashing with bcrypt

### **email-validator**
- **Version**: `email-validator>=2.0.0`
- **Purpose**: Email validation for Pydantic

---

## 💾 Database Packages

### **SQLAlchemy**
- **Version**: `sqlalchemy>=1.4.0`
- **Purpose**: SQL toolkit and ORM

### **Alembic**
- **Version**: `alembic>=1.8.0`
- **Purpose**: Database migration tool

---

## 📄 Report Generation Packages

### **reportlab**
- **Version**: `reportlab>=4.0.0`
- **Purpose**: PDF generation

### **python-docx**
- **Version**: `python-docx>=1.0.0`
- **Purpose**: Word document generation

### **matplotlib**
- **Version**: `matplotlib>=3.8.0`
- **Purpose**: Plotting and visualization

### **seaborn**
- **Version**: `seaborn>=0.13.0`
- **Purpose**: Statistical data visualization

---

## 🔌 API Integration Packages

### **OpenAI**
- **Version**: `openai>=1.3.0`
- **Purpose**: OpenAI API integration for enhanced features

---

## 🛠️ Utility Packages

### **python-dotenv**
- **Version**: `python-dotenv>=1.0.0`
- **Purpose**: Environment variable management

### **aiofiles**
- **Version**: `aiofiles>=23.2.1`
- **Purpose**: Async file I/O

### **jinja2**
- **Version**: `jinja2>=3.1.2`
- **Purpose**: Template engine for HTML/PDF reports

---

## 🧪 Development & Testing

### **pytest**
- **Version**: `pytest>=7.4.0`
- **Purpose**: Testing framework

### **pytest-asyncio**
- **Version**: `pytest-asyncio>=0.21.0`
- **Purpose**: Async test support

---

## 🚀 Deployment Packages

### **docker**
- **Version**: `docker>=6.1.0`
- **Purpose**: Containerization

### **gunicorn**
- **Version**: `gunicorn>=21.2.0`
- **Purpose**: WSGI HTTP server for production

---

## 📦 Frontend Packages (Node.js)

### **React**
- **Version**: `react@^18.2.0`
- **Purpose**: UI framework

### **React DOM**
- **Version**: `react-dom@^18.2.0`
- **Purpose**: React DOM renderer

### **React Router DOM**
- **Version**: `react-router-dom@^6.20.0`
- **Purpose**: Client-side routing

### **Firebase**
- **Version**: `firebase@^12.4.0`
- **Purpose**: Authentication and backend services

### **Framer Motion**
- **Version**: `framer-motion@^12.23.24`
- **Purpose**: Animation library

### **GSAP**
- **Version**: `gsap@^3.12.5`
- **Purpose**: Animation and motion graphics

### **Vite**
- **Version**: `vite@^5.0.8` (dev dependency)
- **Purpose**: Build tool and dev server

### **@vitejs/plugin-react**
- **Version**: `@vitejs/plugin-react@^4.2.1` (dev dependency)
- **Purpose**: React plugin for Vite

---

## ⚠️ Known Compatibility Issues & Workarounds

### 1. **numba + librosa Compatibility**
- **Issue**: `numba.core.errors.TypingError` in `librosa.pyin` pitch tracking
- **Solution**: Disable numba JIT compilation globally via environment variables
- **Files Modified**: 
  - `backend/app.py` (lines 5-14)
  - `backend/core/settings.py` (lines 19-21)
  - `backend/analysis/audio/enhanced_audio_analyzer.py` (lines 5-25)
  - `backend/analysis/audio/audio_analyzer.py` (lines 5-23)

### 2. **numpy + tensorflow Compatibility**
- **Issue**: `numpy>=1.25.0` incompatible with `tensorflow==2.15.1`
- **Solution**: Constrain numpy to `<1.25.0`
- **Note**: This also ensures compatibility with librosa 0.10.x and numba 0.56.x/0.57.x

### 3. **numba + Python 3.11**
- **Issue**: `numba==0.56.x` has issues with Python 3.11
- **Solution**: Use `numba==0.57.1` which supports Python 3.11
- **Note**: Despite this, JIT is still disabled due to librosa compatibility

---

## 📋 Installation Notes

### Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model (required)
python -m spacy download en_core_web_md
```

### Critical Installation Order

If you encounter dependency conflicts:

1. **First**: Install numpy with constraint
   ```bash
   pip install "numpy>=1.21.0,<1.25.0"
   ```

2. **Second**: Install numba
   ```bash
   pip install "numba>=0.56.0,<0.57.0"
   ```

3. **Third**: Install librosa
   ```bash
   pip install "librosa>=0.10.0,<0.11.0"
   ```

4. **Then**: Install rest of dependencies
   ```bash
   pip install -r requirements.txt
   ```

### Node.js Setup

```bash
# Install frontend dependencies
npm install
```

---

## 🔄 Version Update Policy

- **numba**: Do NOT update beyond 0.57.x unless librosa compatibility is verified
- **numpy**: Do NOT update to 1.25.0+ (breaks tensorflow 2.15.x)
- **librosa**: Do NOT update to 0.11.0+ (may break numba compatibility)
- **tensorflow/tf-keras**: Keep at 2.15.x for stability

---

## 📝 Version Verification

To verify installed versions in your environment:

```bash
# Python packages
pip list | grep -E "numba|numpy|librosa|tensorflow"

# Or check specific package
python -c "import numba; import numpy; import librosa; print(f'numba: {numba.__version__}'); print(f'numpy: {numpy.__version__}'); print(f'librosa: {librosa.__version__}')"

# Node packages
npm list
```

---

**Last Updated**: 2026-01-16  
**Python Version**: 3.11+  
**Node Version**: 18+ (recommended)
