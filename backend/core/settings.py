"""
Configuration settings for Face2Phrase
"""
import os
from pathlib import Path

# Project paths - backend/core/settings.py is 2 levels deep from project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_DIR = PROJECT_ROOT / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
MODELS_DIR = PROJECT_ROOT / "models_cache"
REPORTS_DIR = STORAGE_DIR / "reports"
REPORTS_JSON_DIR = REPORTS_DIR / "json"
REPORTS_PDF_DIR = REPORTS_DIR / "pdf"
THUMBNAIL_DIR = REPORTS_DIR / "thumbnails"
EXPORTS_DIR = STORAGE_DIR / "exports"
DB_DIR = STORAGE_DIR / "db"

# Disable numba JIT compilation to avoid compatibility issues
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable numba JIT to prevent compilation errors

# Create directories if they don't exist (in correct order)
STORAGE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
# Create base reports directory first
REPORTS_BASE_DIR = STORAGE_DIR / "reports"
REPORTS_BASE_DIR.mkdir(exist_ok=True)
# Then create subdirectories
REPORTS_JSON_DIR.mkdir(exist_ok=True)
REPORTS_PDF_DIR.mkdir(exist_ok=True)
THUMBNAIL_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)

# Backwards compatibility: REPORTS_DIR points to JSON directory
REPORTS_DIR = REPORTS_JSON_DIR

# File upload settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.aac'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# AI Model settings - HYBRID APPROACH
WHISPER_MODEL_SIZE = "large-v3"  # Accurate transcription (large-v3 for proper nouns)
SPACY_MODEL = "en_core_web_md"
DEEPFACE_MODELS = ["emotion", "age", "gender"]

# Whisper Transcription Quality Settings
WHISPER_LANGUAGE = "en"  # Force English for accuracy
WHISPER_TEMPERATURE = 0.0  # Deterministic transcription
WHISPER_NO_SPEECH_THRESHOLD = 0.4  # Lowered from 0.6 to capture filler sounds like "uh", "ah"
WHISPER_LOGPROB_THRESHOLD = None  # Disabled - keep ALL words for verbatim transcription (filler words like uh, um have low confidence)
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4  # Detect repetitive/hallucinated content

# Analysis thresholds
LONG_PAUSE_THRESHOLD = 2.0  # seconds
MEDIUM_PAUSE_THRESHOLD = 1.0  # seconds (new threshold)
SHORT_PAUSE_THRESHOLD = 0.3  # seconds (reduced from 0.5 for precision)

# Comprehensive Filler Words List (expanded for accurate detection)
FILLER_WORDS = [
    # Basic fillers
    "um", "uh", "ah", "eh", "er", "hmm", "erm",
    # Extended variations
    "umm", "uhh", "ahh", "ehh", "err", "uhm", "ahm",
    "ummm", "uhhh", "ahhh", "ehhh", "errr",
    # Compound fillers
    "um-um", "uh-uh", "ah-ah", "uh-huh", "mm-hmm", "mhm",
    # Common phrase fillers
    "like", "you know", "sort of", "kind of", "actually", 
    "basically", "literally", "totally", "really", "so", "well",
    "i mean", "you see", "right", "okay", "alright",
    # Casual speech patterns
    "kinda", "sorta", "gonna", "wanna", "gotta",
    # Agreement/acknowledgment sounds
    "yeah", "yep", "yup", "uh-huh", "mm", "mmm"
]

# Filler words that indicate mumbling/hesitation (subset for special handling)
MUMBLING_FILLERS = [
    "um", "uh", "ah", "eh", "er", "hmm", "erm",
    "umm", "uhh", "ahh", "ehh", "err", "mhm", "mm", "mmm"
]

# Confidence scoring weights
VOICE_CONFIDENCE_WEIGHT = 0.3
FACIAL_CONFIDENCE_WEIGHT = 0.3
VOCABULARY_WEIGHT = 0.4

# Video processing
FRAME_EXTRACTION_FPS = 1  # Extract 1 frame per second for analysis
MIN_FACE_CONFIDENCE = 0.5  # Minimum confidence for face detection

# ACOUSTIC FILLER DETECTION - DISABLED (causes false positives and duplicates)
DISABLE_ACOUSTIC_FILLER_DETECTION = True  # DISABLED - use only Whisper transcript
ACOUSTIC_FILLER_MIN_CONFIDENCE = 0.0  # Not used when disabled
ACOUSTIC_FILLER_MIN_RMS = 0.005  # Not used when disabled

# Cleanup settings
FILE_CLEANUP_HOURS = 24  # Delete uploaded files after 24 hours

# API Integration Settings - OpenAI Only (Free Tier Available)
USE_OPENAI_API = True  # Enabled - Uses OpenAI free tier
# Load from environment variable - NEVER commit secrets to version control
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Free tier model for cost-effective analysis

# Database Configuration
DATABASE_PATH = DB_DIR / "face2phrase.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"  # SQLite for development

# Authentication
# Load from environment variable - generate secure key: python -c "import secrets; print(secrets.token_urlsafe(32))"
import secrets
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Rate Limiting (configured but not yet implemented)
RATE_LIMIT_PER_MINUTE = 60  # API rate limiting
MAX_CONCURRENT_PROCESSES = 5  # Parallel processing limit

# Validation Settings (for accurate scoring)
MIN_TRANSCRIPTION_CONFIDENCE = 0.0  # Disabled - keep ALL words including filler sounds
MIN_METRIC_CONFIDENCE = 0.65  # Show "Insufficient Data" if below this
VALIDATION_ENABLED = True  # Global flag for validation checks

# Speaking Rate Categories (based on research)
WPM_SLOW_THRESHOLD = 125  # Below this is considered slow
WPM_FAST_THRESHOLD = 175  # Above this is considered fast
WPM_MIN_PLAUSIBLE = 30  # Minimum physically possible
WPM_MAX_PLAUSIBLE = 300  # Maximum physically possible
