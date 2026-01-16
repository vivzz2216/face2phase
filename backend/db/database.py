"""
Database Models and User Management
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional, List
import json
from ..core.settings import DATABASE_URL

Base = declarative_base()

class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="user")
    session_summaries = relationship("SessionSummary", back_populates="user")

class Analysis(Base):
    """Analysis results storage"""
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String(100), unique=True, index=True)
    file_name = Column(String(255))
    file_type = Column(String(20))  # 'audio' or 'video'
    file_size = Column(Integer)
    
    # Analysis results (stored as JSON)
    audio_analysis = Column(Text)  # JSON string
    facial_analysis = Column(Text)  # JSON string
    text_analysis = Column(Text)  # JSON string
    overall_score = Column(Float)
    
    # Metadata
    processing_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analyses")


class SessionSummary(Base):
    """Summarized session history for fast retrieval"""
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(255))
    file_name = Column(String(255))
    file_type = Column(String(20))
    overall_score = Column(Float)
    score_breakdown = Column(Text)  # JSON
    highlights = Column(Text)  # JSON
    metrics = Column(Text)  # JSON
    pdf_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="session_summaries")

class Analytics(Base):
    """Analytics and usage statistics"""
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    metric_name = Column(String(100))
    metric_value = Column(Float)
    metric_data = Column(Text)  # JSON string for complex data
    recorded_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    """Database management class"""
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_analysis(self, user_id: int, session_id: str, analysis_data: dict):
        """Save analysis results to database"""
        session = self.get_session()
        try:
            analysis = Analysis(
                user_id=user_id,
                session_id=session_id,
                file_name=analysis_data.get('file_name', ''),
                file_type=analysis_data.get('file_type', ''),
                file_size=analysis_data.get('file_size', 0),
                audio_analysis=json.dumps(analysis_data.get('audio_analysis', {})),
                facial_analysis=json.dumps(analysis_data.get('facial_analysis', {})),
                text_analysis=json.dumps(analysis_data.get('text_analysis', {})),
                overall_score=analysis_data.get('overall_score', 0),
                processing_time=analysis_data.get('processing_time', 0)
            )
            session.add(analysis)
            session.commit()
            return analysis.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_user_analyses(self, user_id: int, limit: int = 1000):
        """Get user's analysis history - returns ALL analyses (high limit for logged-in users)"""
        session = self.get_session()
        try:
            # Use high limit to get all analyses for logged-in users
            analyses = session.query(Analysis).filter(
                Analysis.user_id == user_id
            ).order_by(Analysis.created_at.desc()).limit(limit).all()
            
            results = []
            for analysis in analyses:
                audio_data = json.loads(analysis.audio_analysis) if analysis.audio_analysis else {}
                facial_data = json.loads(analysis.facial_analysis) if analysis.facial_analysis else {}
                text_data = json.loads(analysis.text_analysis) if analysis.text_analysis else {}
                
                results.append({
                    'id': analysis.id,
                    'session_id': analysis.session_id,
                    'file_name': analysis.file_name,
                    'file_type': analysis.file_type,
                    'overall_score': analysis.overall_score,
                    'voice_confidence': audio_data.get('voice_confidence_score', 0),
                    'facial_confidence': facial_data.get('facial_confidence_score', 0),
                    'vocabulary_score': text_data.get('vocabulary_score', 0),
                    'created_at': analysis.created_at.isoformat(),
                    'audio_analysis': audio_data,
                    'facial_analysis': facial_data,
                    'text_analysis': text_data
                })
            return results
        finally:
            session.close()
    
    def get_analysis_by_session(self, session_id: str):
        """Get analysis by session ID"""
        session = self.get_session()
        try:
            analysis = session.query(Analysis).filter(
                Analysis.session_id == session_id
            ).first()
            
            if analysis:
                return {
                    'id': analysis.id,
                    'user_id': analysis.user_id,
                    'session_id': analysis.session_id,
                    'file_name': analysis.file_name,
                    'file_type': analysis.file_type,
                    'overall_score': analysis.overall_score,
                    'created_at': analysis.created_at.isoformat(),
                    'audio_analysis': json.loads(analysis.audio_analysis) if analysis.audio_analysis else {},
                    'facial_analysis': json.loads(analysis.facial_analysis) if analysis.facial_analysis else {},
                    'text_analysis': json.loads(analysis.text_analysis) if analysis.text_analysis else {}
                }
            return None
        finally:
            session.close()

    def save_session_summary(self, summary: dict):
        """Insert or update a session summary record."""
        session = self.get_session()
        try:
            existing = session.query(SessionSummary).filter(
                SessionSummary.session_id == summary['session_id']
            ).first()

            if existing:
                existing.title = summary.get('title', existing.title)
                existing.file_name = summary.get('file_name', existing.file_name)
                existing.file_type = summary.get('file_type', existing.file_type)
                existing.overall_score = summary.get('overall_score', existing.overall_score)
                existing.score_breakdown = json.dumps(summary.get('score_breakdown', {}))
                existing.highlights = json.dumps(summary.get('highlights', {}))
                existing.metrics = json.dumps(summary.get('metrics', {}))
                existing.pdf_path = summary.get('pdf_path', existing.pdf_path)
                if summary.get('user_id') and not existing.user_id:
                    existing.user_id = summary['user_id']
            else:
                new_summary = SessionSummary(
                    user_id=summary.get('user_id'),
                    session_id=summary['session_id'],
                    title=summary.get('title'),
                    file_name=summary.get('file_name'),
                    file_type=summary.get('file_type'),
                    overall_score=summary.get('overall_score'),
                    score_breakdown=json.dumps(summary.get('score_breakdown', {})),
                    highlights=json.dumps(summary.get('highlights', {})),
                    metrics=json.dumps(summary.get('metrics', {})),
                    pdf_path=summary.get('pdf_path')
                )
                session.add(new_summary)

            session.commit()
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()

    def get_user_sessions(self, user_id: int, limit: int = 50):
        """Return summarized sessions for a user."""
        session = self.get_session()
        try:
            summaries = (
                session.query(SessionSummary)
                .filter(SessionSummary.user_id == user_id)
                .order_by(SessionSummary.created_at.desc())
                .limit(limit)
                .all()
            )

            results = []
            for summary in summaries:
                score_breakdown = json.loads(summary.score_breakdown) if summary.score_breakdown else {}
                highlights = json.loads(summary.highlights) if summary.highlights else {}
                metrics = json.loads(summary.metrics) if summary.metrics else {}
                results.append({
                    "id": summary.id,
                    "session_id": summary.session_id,
                    "title": summary.title,
                    "file_name": summary.file_name,
                    "file_type": summary.file_type,
                    "overall_score": summary.overall_score,
                    "score_breakdown": score_breakdown,
                    "highlights": highlights,
                    "metrics": metrics,
                    "pdf_path": summary.pdf_path,
                    "created_at": summary.created_at.isoformat() if summary.created_at else None,
                    "updated_at": summary.updated_at.isoformat() if summary.updated_at else None,
                    # Backwards compatibility fields for existing UI
                    "voice_confidence": score_breakdown.get("voice_confidence"),
                    "facial_confidence": score_breakdown.get("facial_confidence"),
                    "vocabulary_score": score_breakdown.get("vocabulary_score"),
                    "duration": metrics.get("duration"),
                    "thumbnail_url": highlights.get("thumbnail_url"),
                })
            return results
        finally:
            session.close()

    def get_session_summary(self, session_id: str):
        """Fetch single session summary by session ID."""
        session = self.get_session()
        try:
            summary = session.query(SessionSummary).filter(
                SessionSummary.session_id == session_id
            ).first()
            if not summary:
                return None

            return {
                "id": summary.id,
                "user_id": summary.user_id,
                "session_id": summary.session_id,
                "title": summary.title,
                "file_name": summary.file_name,
                "file_type": summary.file_type,
                "overall_score": summary.overall_score,
                "score_breakdown": json.loads(summary.score_breakdown) if summary.score_breakdown else {},
                "highlights": json.loads(summary.highlights) if summary.highlights else {},
                "metrics": json.loads(summary.metrics) if summary.metrics else {},
                "pdf_path": summary.pdf_path,
                "created_at": summary.created_at.isoformat() if summary.created_at else None,
                "updated_at": summary.updated_at.isoformat() if summary.updated_at else None
            }
        finally:
            session.close()

    def delete_session(self, session_id: str, user_id: Optional[int] = None) -> bool:
        """
        Delete a session summary and associated analysis rows.

        Returns True if a record was removed, False if nothing matched or user mismatch.
        """
        session = self.get_session()
        try:
            deleted = False

            summary = session.query(SessionSummary).filter(SessionSummary.session_id == session_id).first()
            if summary:
                if summary.user_id and user_id and summary.user_id != user_id:
                    return False
                session.delete(summary)
                deleted = True

            analysis = session.query(Analysis).filter(Analysis.session_id == session_id).first()
            if analysis:
                if analysis.user_id and user_id and analysis.user_id != user_id:
                    session.rollback()
                    return False
                session.delete(analysis)
                deleted = True

            session.commit()
            return deleted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_all_sessions(self, user_id: int) -> List[str]:
        """
        Delete all sessions for a given user.

        Returns list of session IDs removed.
        """
        session = self.get_session()
        deleted: List[str] = []
        try:
            summaries = session.query(SessionSummary).filter(SessionSummary.user_id == user_id).all()
            analyses = session.query(Analysis).filter(Analysis.user_id == user_id).all()

            for summary in summaries:
                deleted.append(summary.session_id)
                session.delete(summary)

            for analysis in analyses:
                if analysis.session_id not in deleted:
                    deleted.append(analysis.session_id)
                session.delete(analysis)

            session.commit()
            return deleted
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

# Global database manager instance
db_manager = DatabaseManager()
