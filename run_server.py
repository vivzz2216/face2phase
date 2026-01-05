"""
Startup script for Face2Phase server
Adds backend directory to Python path before starting uvicorn
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server on http://127.0.0.1:8000...")
    try:
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=[str(backend_dir)]
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
    print("Server stopped.")
