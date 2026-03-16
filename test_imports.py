"""
Test script to verify all required dependencies are installed
"""
import sys

def test_imports():
    """Test all critical imports"""
    failed = []
    warnings = []
    
    print("Testing critical dependencies...\n")
    
    # Core dependencies
    tests = [
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("Pydantic", "pydantic"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Librosa", "librosa"),
        ("SoundFile", "soundfile"),
        ("Whisper", "whisper"),
        ("OpenCV", "cv2"),
        ("MTCNN", "mtcnn"),
        ("DeepFace", "deepface"),
        ("TensorFlow", "tensorflow"),
        ("tf-keras", "tf_keras"),
        ("spaCy", "spacy"),
        ("NLTK", "nltk"),
        ("Transformers", "transformers"),
        ("Pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("MoviePy", "moviepy"),
        ("python-dotenv", "dotenv"),
        ("aiofiles", "aiofiles"),
        ("Jinja2", "jinja2"),
        ("python-jose", "jose"),
        ("Passlib", "passlib"),
        ("SQLAlchemy", "sqlalchemy"),
        ("Alembic", "alembic"),
        ("OpenAI", "openai"),
        ("ReportLab", "reportlab"),
        ("python-docx", "docx"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
    ]
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    # Optional dependencies
    print("\nTesting optional dependencies...\n")
    optional = [
        ("webrtcvad", "webrtcvad"),
    ]
    
    for name, module in optional:
        try:
            __import__(module)
            print(f"✅ {name} (optional)")
        except ImportError:
            print(f"⚠️  {name} (optional - not installed)")
            warnings.append(name)
    
    print("\n" + "="*50)
    if failed:
        print(f"\n❌ FAILED: {len(failed)} critical packages missing:")
        for pkg in failed:
            print(f"   - {pkg}")
        return False
    else:
        print("\n✅ SUCCESS: All critical dependencies installed!")
        if warnings:
            print(f"\n⚠️  {len(warnings)} optional packages not installed:")
            for pkg in warnings:
                print(f"   - {pkg}")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
