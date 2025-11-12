"""
Export job manager for handling asynchronous report generation.
"""
import threading
import uuid
from typing import Dict, Optional


class ExportJobManager:
    """Tracks export job state for PDF/other report generations."""

    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create_job(self, session_id: str, export_type: str) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "session_id": session_id,
                "export_type": export_type,
                "status": "pending",
                "message": "Queued",
                "file_path": None
            }
        return job_id

    def update_job(self, job_id: str, **fields) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(fields)

    def fail_job(self, job_id: str, message: str) -> None:
        self.update_job(job_id, status="failed", message=message)

    def complete_job(self, job_id: str, file_path: str) -> None:
        self.update_job(job_id, status="completed", message="Export ready", file_path=file_path)

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                return dict(job)
        return None


# Global instance
export_job_manager = ExportJobManager()

