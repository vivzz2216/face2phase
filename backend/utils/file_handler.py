"""
File upload and validation utilities
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import Optional, Tuple
import logging
from ..core.settings import (
    UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXTENSIONS,
    ALLOWED_AUDIO_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS
)

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file uploads, validation, and cleanup"""
    
    def __init__(self):
        self.upload_dir = UPLOAD_DIR
        self.upload_dir.mkdir(exist_ok=True)
    
    def validate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to the uploaded file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return False, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        
        # Check file extension
        file_ext = file_path.suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return False, f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        
        return True, ""
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Tuple[Optional[Path], str]:
        """
        Save uploaded file to disk
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Tuple of (saved_file_path, error_message)
        """
        try:
            # Generate unique session ID and file path
            session_id = str(uuid.uuid4())
            file_ext = Path(filename).suffix.lower()
            saved_filename = f"{session_id}{file_ext}"
            file_path = self.upload_dir / saved_filename
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Validate saved file
            is_valid, error_msg = self.validate_file(file_path)
            if not is_valid:
                file_path.unlink()  # Delete invalid file
                return None, error_msg
            
            logger.info(f"File saved: {file_path}")
            return file_path, ""
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return None, f"Error saving file: {str(e)}"
    
    def get_file_type(self, file_path: Path) -> str:
        """
        Determine if file is audio or video
        
        Args:
            file_path: Path to the file
            
        Returns:
            'audio', 'video', or 'unknown'
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext in ALLOWED_AUDIO_EXTENSIONS:
            return "audio"
        elif file_ext in ALLOWED_VIDEO_EXTENSIONS:
            return "video"
        else:
            return "unknown"
    
    def cleanup_file(self, file_path: Path) -> bool:
        """
        Delete file from disk
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def cleanup_session_files(self, session_id: str) -> bool:
        """
        Clean up all files for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all files with this session ID
            pattern = f"{session_id}*"
            files_to_delete = list(self.upload_dir.glob(pattern))
            
            success = True
            for file_path in files_to_delete:
                if not self.cleanup_file(file_path):
                    success = False
            
            return success
        except Exception as e:
            logger.error(f"Error cleaning up session files: {e}")
            return False

# Global file handler instance
file_handler = FileHandler()
