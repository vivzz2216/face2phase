"""
Configuration settings for Face2Phrase
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MODELS_DIR = PROJECT_ROOT / "models_cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
THUMBNAIL_DIR = REPORTS_DIR / "thumbnails"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)

# File upload settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# AI Model settings
WHISPER_MODEL_SIZE = "medium"  # Options: tiny, base, small, medium, large (upgraded from base for better accuracy)
SPACY_MODEL = "en_core_web_md"
DEEPFACE_MODELS = ["emotion", "age", "gender"]

# Analysis thresholds
LONG_PAUSE_THRESHOLD = 2.0  # seconds
SHORT_PAUSE_THRESHOLD = 0.5  # seconds
FILLER_WORDS = [
    "um", "uh", "like", "you know", "sort of", "actually", 
    "basically", "literally", "totally", "really", "so", "well"
]

# Confidence scoring weights
VOICE_CONFIDENCE_WEIGHT = 0.3
FACIAL_CONFIDENCE_WEIGHT = 0.3
VOCABULARY_WEIGHT = 0.4

# Video processing
FRAME_EXTRACTION_FPS = 1  # Extract 1 frame per second for analysis
MIN_FACE_CONFIDENCE = 0.5  # Minimum confidence for face detection

# Cleanup settings
FILE_CLEANUP_HOURS = 24  # Delete uploaded files after 24 hours

# API Integration Settings - OpenAI Only (Free Tier Available)
USE_OPENAI_API = True  # Enabled - Uses OpenAI free tier
# Load from environment variable - NEVER commit secrets to version control
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Free tier model for cost-effective analysis

# Database Configuration
DATABASE_URL = "sqlite:///face2phrase.db"  # SQLite for development

# Authentication
# Load from environment variable - generate secure key: python -c "import secrets; print(secrets.token_urlsafe(32))"
import secrets
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Rate Limiting (configured but not yet implemented)
RATE_LIMIT_PER_MINUTE = 60  # API rate limiting
MAX_CONCURRENT_PROCESSES = 5  # Parallel processing limit

# Score Boost Setting (adds points to all analysis scores)
SCORE_BOOST = 30  # Add 30 points to every analysis score
