"""
FastAPI main application with upload, processing, and report endpoints
"""
# CRITICAL FIX: Disable numba JIT BEFORE any imports to fix librosa compatibility
import os
# Disable numba JIT compilation completely to avoid compatibility issues with librosa
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_DISABLE_CUDA'] = '1'
# Try to disable numba programmatically as well
try:
    import numba
    numba.config.DISABLE_JIT = True
except (ImportError, AttributeError):
    pass  # Numba not available or config not accessible

# Try to load dotenv, fallback if not available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file
except ImportError:
    print("dotenv not available, using environment variables directly")
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional
import subprocess

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Header, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn

from .analysis.audio.audio_analyzer import audio_analyzer
from .analysis.audio.enhanced_audio_analyzer import enhanced_audio_analyzer
from .exporters.pdf_report_generator import pdf_word_generator
from .analysis.face.facial_analyzer import facial_analyzer
from .analysis.text.text_analyzer import text_analyzer
from .exporters.report_generator import report_generator
from .services.openai_enhancer import openai_enhancer
from .analysis.audio.strict_audio_evaluator import StrictAudioEvaluator
from .db.database import db_manager, User
from .services.auth import auth_manager
from .utils.file_handler import file_handler
from .utils.device_detector import device_manager
from .services.detailed_feedback_generator import detailed_feedback_generator
from .services.chatbot_service import chatbot_service
from .exporters.pro_pdf_exporter import pro_pdf_exporter
from .exporters.export_job_manager import export_job_manager
from .analysis.text.transcript_processor import transcript_processor
from .analysis.text.word_analyzer import word_analyzer
from .analysis.speech.pronunciation_analyzer import pronunciation_analyzer
from .analysis.text.transcript_enhancer import transcript_enhancer
from .analysis.text.vocabulary_enhancer import vocabulary_enhancer
from .core.settings import UPLOAD_DIR, REPORTS_DIR, THUMBNAIL_DIR, USE_OPENAI_API, MAX_FILE_SIZE, ALLOWED_EXTENSIONS
from .utils.video_utils import extract_audio_from_video, generate_video_thumbnail

# Ensure stdout/stderr can emit UTF-8 (important for non-ASCII transcripts/logs)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logging with detailed debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('face2phrase.log', encoding='utf-8')
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face2Phrase",
    description="Offline AI Communication Feedback System",
    version="1.0.0"
)

@app.exception_handler(ConnectionResetError)
async def handle_connection_reset(request: Request, exc: ConnectionResetError):
    """
    Swallow client disconnect errors (WinError 10054) that occur when browsers cancel
    range/streaming requests mid-transfer. Returning a 499 keeps logs clean while
    preventing noisy stack traces.
    """
    logger.debug(f"Client disconnected: {request.url} - {exc}")
    return Response(status_code=499)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:5174", 
        "http://localhost:8000",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:3001", 
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Global processing status tracking
processing_status = {}

# Mount static files and templates (after routes to avoid conflicts)
try:
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

try:
    app.mount("/thumbnails", StaticFiles(directory=THUMBNAIL_DIR), name="thumbnails")
except Exception as e:
    logger.warning(f"Could not mount thumbnails directory: {e}")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

async def process_file_async(session_id: str, file_path: Path, file_type: str):
    """
    Process uploaded file asynchronously
    
    Args:
        session_id: Session identifier
        file_path: Path to uploaded file
        file_type: Type of file ('audio' or 'video')
    """
    try:
        logger.info(f"Starting processing for session: {session_id}")
        logger.debug(f"File path: {file_path}")
        logger.debug(f"File type: {file_type}")
        logger.debug(f"File exists: {file_path.exists()}")
        logger.debug(f"File size: {file_path.stat().st_size if file_path.exists() else 'N/A'}")
        
        current_status = processing_status.get(session_id, {})
        current_status.update({
            "status": "processing",
            "progress": 0,
            "file_path": str(file_path),
            "file_type": file_type,
            "filename": current_status.get("filename"),
            "user_id": current_status.get("user_id"),
            "file_size": current_status.get("file_size"),
        })
        processing_status[session_id] = current_status
        
        # Extract audio if video
        audio_path = file_path
        thumbnail_url = None
        if file_type == "video":
            logger.info("Processing video file - extracting audio")
            if session_id in processing_status:
                processing_status[session_id]["progress"] = 10
            audio_path = extract_audio_from_video(file_path)
            if not audio_path:
                raise Exception("Failed to extract audio from video")
            logger.info(f"Audio extracted to: {audio_path}")
        
        # Run analyses (audio + visual) concurrently where possible
        if session_id in processing_status:
            processing_status[session_id]["progress"] = 20
        logger.info("Starting analysis pipeline (audio + visual parallel)")

        audio_task = asyncio.to_thread(enhanced_audio_analyzer.analyze_audio_comprehensive, audio_path)
        facial_task = None
        if file_type == "video":
            logger.info("Starting facial analysis (parallel)")
            facial_task = asyncio.to_thread(facial_analyzer.analyze_video, file_path)

        audio_results = {"analysis_successful": False}
        try:
            audio_results = await audio_task
            logger.debug(f"Audio analysis results keys: {list(audio_results.keys())}")
            if session_id in processing_status:
                processing_status[session_id]['audio_results'] = audio_results
            
            transcript = audio_results.get('transcript', '')
            enhanced_transcript = audio_results.get('enhanced_transcript', '')
            
            logger.info("Audio analysis completed")
            logger.info(f"Transcript length: {len(transcript) if isinstance(transcript, str) else 'N/A'}")
            logger.info(f"Enhanced transcript length: {len(enhanced_transcript)}")
            
            # DEBUG: Log transcript preview
            if transcript:
                logger.info(f"📝 Transcript preview (first 100 chars): {transcript[:100]}")
            else:
                logger.warning("⚠️ TRANSCRIPT IS EMPTY! This will cause issues in the UI.")
                logger.warning(f"Audio results keys: {list(audio_results.keys())}")

            pause_summary = audio_results.get('pause_summary', {})
            logger.info(f"Pauses detected: {pause_summary.get('total_pauses', 0)}")
            filler_analysis = audio_results.get('filler_analysis', {})
            logger.info(f"Filler words detected: {filler_analysis.get('total_fillers', 0)}")
        except Exception as e:
            logger.error(f"Enhanced audio analysis failed: {e}")
            audio_results = {"analysis_successful": False, "error": str(e)}
        
        if session_id in processing_status:
            processing_status[session_id]["progress"] = 50
        else:
            logger.warning(f"Session {session_id} missing during progress update (50%)")
        
        facial_results = {"analysis_successful": True, "facial_confidence_score": 0}
        if facial_task:
            try:
                facial_results = await facial_task
                logger.debug(f"Facial analysis results keys: {list(facial_results.keys())}")
                logger.info("Facial analysis completed")
            except Exception as e:
                logger.error(f"Facial analysis failed: {e}")
                facial_results = {"analysis_successful": False, "error": str(e)}
        
        if session_id in processing_status:
            processing_status[session_id]["progress"] = 70
        
        # Word analysis for weak words, fillers, and vocabulary (run in parallel)
        word_analysis_results = {}
        transcript_improvement = {}
        vocabulary_enhancements = {}
        transcript = audio_results.get('transcript', '') or audio_results.get('enhanced_transcript', '')

        if transcript:
            logger.info("Starting text enhancement suite (parallel)")
            text_tasks = [
                asyncio.to_thread(word_analyzer.get_comprehensive_analysis, transcript),
                asyncio.to_thread(transcript_enhancer.enhance_transcript, transcript),
                asyncio.to_thread(vocabulary_enhancer.enhance_vocabulary, transcript)
            ]

            word_result, transcript_result, vocab_result = await asyncio.gather(
                *text_tasks, return_exceptions=True
            )

            if isinstance(word_result, Exception):
                logger.error(f"Word analysis failed: {word_result}")
            else:
                word_analysis_results = word_result or {}
                logger.info(f"Word analysis completed: {word_analysis_results.get('summary', {})}")
                
            if isinstance(transcript_result, Exception):
                logger.error(f"Transcript enhancement failed: {transcript_result}")
            else:
                transcript_improvement = transcript_result or {}
                logger.info(f"Transcript enhancement completed: {transcript_improvement.get('success', False)}")
                
            if isinstance(vocab_result, Exception):
                logger.error(f"Vocabulary enhancement failed: {vocab_result}")
            else:
                vocabulary_enhancements = vocab_result or {}
                logger.info(f"Vocabulary enhancement completed: {vocabulary_enhancements.get('total_suggestions', 0)} suggestions")
        else:
            logger.info("Transcript unavailable; skipping text enhancement suite.")
        
        # Use STRICT audio-only evaluator for audio files
        logger.info("Starting STRICT audio-only evaluation")
        logger.info(f"AUDIO RESULTS STRUCTURE: {list(audio_results.keys())}")
        
        # Get words with timing for strict evaluation
        words_with_timing = audio_results.get('words_with_timing', [])
        duration_sec = audio_results.get('speaking_metrics', {}).get('total_duration', 0)
        
        # Fallback to audio file duration if not in metrics
        if duration_sec == 0:
            import librosa
            try:
                audio_data, sr = librosa.load(str(audio_path), sr=16000)
                duration_sec = len(audio_data) / sr
            except:
                duration_sec = 0
        
        try:
            # Initialize strict evaluator
            strict_evaluator = StrictAudioEvaluator()
            
            # Run strict evaluation
            strict_eval_result = strict_evaluator.evaluate(
                words_with_timing, 
                duration_sec, 
                filler_stats=audio_results.get('filler_analysis', {}),
                pause_stats=audio_results.get('pause_summary', {})
            )
            
            logger.info("STRICT EVALUATION RESULTS:")
            logger.info(f"Final score: {strict_eval_result['scores']['final_100']}")
            logger.info(f"Flags: {strict_eval_result['flags']}")
            logger.info(f"Issues: {strict_eval_result['top_issues']}")
            
            # Format as text_results compatible format
            text_results = {
                'vocabulary_metrics': {
                    'total_words': strict_eval_result['metrics']['words'],
                    'unique_words': int(strict_eval_result['metrics']['words'] * strict_eval_result['metrics']['lexical_diversity_ttr']),
                    'vocabulary_richness': strict_eval_result['metrics']['lexical_diversity_ttr']
                },
                'structure_metrics': {},
                'content_metrics': {},
                'vocabulary_score': ((strict_eval_result['scores']['coherence_grammar_25'] + strict_eval_result['scores']['content_accuracy_20']) / 45.0) * 100,
                'summary': f"Strict audio-only evaluation. Score: {strict_eval_result['scores']['final_100']}/100",
                'analysis_successful': True,
                'strict_evaluation': strict_eval_result
            }
            
            # Override voice confidence with strict clarity score
            if 'voice_confidence_score' in audio_results:
                # Use strict clarity + fluency scores for voice
                audio_results['voice_confidence_score'] = (
                    (strict_eval_result['scores']['clarity_pronunciation_25'] + 
                     strict_eval_result['scores']['fluency_pace_20']) / 45.0
                ) * 100
            
            logger.info("Strict evaluation completed")
            
        except Exception as e:
            logger.error(f"Strict evaluation failed, falling back to standard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback to standard text analysis
            transcript = audio_results.get('transcript', '')
            if not transcript or len(transcript) == 0:
                text_results = {
                    'vocabulary_metrics': {},
                    'structure_metrics': {},
                    'content_metrics': {},
                    'vocabulary_score': 0,
                    'summary': 'No transcript available.',
                    'analysis_successful': True
                }
            else:
                text_results = text_analyzer.analyze_text(transcript)
        
        if session_id in processing_status:
            processing_status[session_id]["progress"] = 90
        
        # Generate report
        logger.info("Generating final report")
        try:
            report = report_generator.generate_report(
                session_id, audio_results, facial_results, text_results, file_type, 
                word_analysis_results, transcript_improvement, vocabulary_enhancements
            )

            if file_type == "video":
                try:
                    thumbnail_target = THUMBNAIL_DIR / f"{session_id}.jpg"
                    generated = generate_video_thumbnail(file_path, thumbnail_target)
                    if generated and generated.exists():
                        thumbnail_url = f"/thumbnails/{generated.name}"
                        if session_id in processing_status:
                            processing_status[session_id]["thumbnail_url"] = thumbnail_url
                        report["thumbnail_url"] = thumbnail_url
                        logger.info("Thumbnail generated for session %s at %s", session_id, generated)
                    else:
                        logger.warning("Thumbnail not generated for session %s", session_id)
                except Exception as thumb_exc:
                    logger.warning("Thumbnail generation failed for session %s: %s", session_id, thumb_exc)
            
            # Add confidence scores to report
            from .services.confidence_scorer import confidence_scorer
            report = confidence_scorer.add_confidence_scores(report, has_video=(file_type == "video"))
            
            # Perform cross-validation between modules
            from .services.cross_validator import cross_validator
            validation_results = cross_validator.validate_all(
                audio_results,
                facial_results,
                text_results,
                has_video=(file_type == "video")
            )
            report['cross_validation'] = validation_results
            
            logger.debug(f"Report keys: {list(report.keys())}")
            logger.info("Report generation completed")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report = {
                "session_id": session_id,
                "error": str(e),
                "analysis_successful": False
            }
        
        if session_id in processing_status:
            processing_status[session_id]["progress"] = 100
            processing_status[session_id]["status"] = "completed"
            processing_status[session_id]["report"] = report
        
        # Save analysis to database ONLY if user_id is available (logged-in users only)
        # Guests will NOT have their analyses saved
        server_status_entry = processing_status.get(session_id, {})
        user_id = server_status_entry.get("user_id")
        logger.info(f"Processing completed for session_id={session_id}, user_id={user_id}")
        if user_id:
            try:
                analysis_data = {
                    "file_name": server_status_entry.get("filename", ""),
                    "file_type": file_type,
                    "file_size": server_status_entry.get("file_size", 0),
                    "audio_analysis": audio_results,
                    "facial_analysis": facial_results,
                    "text_analysis": text_results,
                    "overall_score": report.get("overall_score", 0),
                    "processing_time": 0  # Could calculate if needed
                }
                db_manager.save_analysis(user_id, session_id, analysis_data)
                logger.info(f"Saved analysis to database for logged-in user_id: {user_id}, session: {session_id}")
                
                try:
                    summary_metadata = {
                        "user_id": user_id,
                        "file_name": server_status_entry.get("filename", ""),
                        "file_type": file_type,
                        "pdf_path": None
                    }
                    session_summary = report_generator.build_session_summary(
                        session_id=session_id,
                        report=report,
                        audio_results=audio_results or {},
                        facial_results=facial_results or {},
                        text_results=text_results or {},
                        metadata={**summary_metadata, "thumbnail_url": thumbnail_url}
                    )
                    db_manager.save_session_summary(session_summary)
                    logger.info(f"Session summary stored for user_id {user_id}, session {session_id}")
                except Exception as summary_error:
                    logger.warning(f"Failed to save session summary: {summary_error}")
            except Exception as e:
                logger.warning(f"Failed to save analysis to database: {e}")
        else:
            logger.info(f"Analysis NOT saved to database (guest user), session: {session_id}")
        
        # Cleanup temporary files
        if file_type == "video" and audio_path and audio_path != file_path:
            try:
                audio_path.unlink()
                logger.info(f"Cleaned up temporary audio file: {audio_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary file: {e}")
        
        logger.info(f"Processing completed successfully for session: {session_id}")
        
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        processing_status[session_id] = {
            "status": "error",
            "error": str(e),
            "progress": 0
        }

class UploadUserInfo(BaseModel):
    email: Optional[str] = None
    uid: Optional[str] = None
    display_name: Optional[str] = None


class DeleteSessionRequest(BaseModel):
    email: Optional[str] = None
    uid: Optional[str] = None


def get_or_create_user(user_email: Optional[str], user_uid: Optional[str], user_display_name: Optional[str]) -> Optional[int]:
    """Find or create a database user matching Firebase credentials."""
    if not user_email:
        logger.warning("get_or_create_user called without email")
        return None

    session = db_manager.get_session()
    try:
        logger.debug(f"Looking up user with email={user_email}, uid={user_uid[:8] if user_uid else None}...")
        user = session.query(User).filter(User.email == user_email).first()
        if not user:
            logger.info(f"User not found for email={user_email}, creating new user")
            username = user_email.split('@')[0]
            existing_username = session.query(User).filter(User.username == username).first()
            if existing_username:
                suffix = user_uid[:8] if user_uid else uuid.uuid4().hex[:8]
                username = f"user_{suffix}"

            user = User(
                username=username,
                email=user_email,
                hashed_password="",
                full_name=user_display_name or username,
                is_active=True
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"Created new backend user: id={user.id}, email={user_email}, username={username}")
        else:
            logger.debug(f"Found existing user: id={user.id}, email={user.email}, username={user.username}")
        return user.id
    except Exception as ex:
        session.rollback()
        logger.error(f"Could not associate Firebase user email={user_email}, uid={user_uid[:8] if user_uid else None}...: {ex}", exc_info=True)
        return None
    finally:
        session.close()

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    email: Optional[str] = Form(None),
    uid: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    user_email: Optional[str] = Form(None),
    user_uid: Optional[str] = Form(None),
    user_display_name: Optional[str] = Form(None),
    project_name: Optional[str] = Form(None),
):
    """Handle file upload and start processing"""
    logger.info(f"Upload endpoint called, filename: {file.filename if file else 'None'}")
    try:
        # Validate file
        if not file or not file.filename:
            logger.warning("Upload attempted without file")
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Validate file extension before reading
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size in memory
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Save file
        file_path, error = file_handler.save_uploaded_file(content, file.filename)
        if not file_path:
            raise HTTPException(status_code=400, detail=error)
        
        # Determine file type
        file_type = file_handler.get_file_type(file_path)
        if file_type == "unknown":
            file_handler.cleanup_file(file_path)
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Get or create user if Firebase info provided
        resolved_email = user_email or email
        resolved_uid = user_uid or uid
        resolved_display_name = user_display_name or display_name

        # DEBUG: Log FormData fields received
        logger.info(f"[UPLOAD DEBUG] FormData - email={email}, uid={uid[:8] if uid else None}..., user_email={user_email}, user_uid={user_uid[:8] if user_uid else None}...")
        logger.info(f"[UPLOAD DEBUG] Resolved - email={resolved_email}, uid={resolved_uid[:8] if resolved_uid else None}...")

        user_id = None
        if resolved_email:
            user_id = await asyncio.to_thread(
                get_or_create_user,
                resolved_email,
                resolved_uid,
                resolved_display_name,
            )
            logger.info(f"[UPLOAD DEBUG] get_or_create_user returned user_id={user_id} for email={resolved_email}")
        else:
            logger.warning(f"[UPLOAD DEBUG] No email provided - guest upload, user_id will be None")
        
        # Store file path and user_id in processing status for later retrieval
        processing_status[session_id] = {
            "status": "uploaded",
            "file_path": str(file_path),
            "filename": file.filename,
            "file_type": file_type,
            "user_id": user_id,
            "file_size": len(content)
        }

        # ----------------------------------------
        # CREATE MINIMAL SESSION SUMMARY ON UPLOAD
        # ----------------------------------------
        if user_id:
            try:
                db_manager.save_session_summary({
                    "user_id": user_id,
                    "session_id": session_id,
                    "title": file.filename,      # for UI
                    "file_name": file.filename,
                    "file_type": file_type,
                    "overall_score": None,
                    "score_breakdown": {},
                    "highlights": {},
                    "metrics": {},
                    "pdf_path": None
                })
                logger.info(f"Created initial session summary for session_id={session_id}, user_id={user_id}, email={resolved_email}")
            except Exception as exc:
                logger.error(f"Failed to create initial session summary for session_id={session_id}, user_id={user_id}: {exc}", exc_info=True)
        else:
            logger.info(f"No user_id available for session_id={session_id} (guest user), email={resolved_email}")

        
        
        # Start background processing
        background_tasks.add_task(process_file_async, session_id, file_path, file_type)
        
        logger.info(f"File uploaded successfully: {file.filename}, session: {session_id}, user_id: {user_id}")
        
        return JSONResponse({
            "session_id": session_id,
            "file_type": file_type,
            "filename": file.filename,
            "message": "File uploaded successfully. Processing started."
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed. Please try again.")


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page"""
    device_info = device_manager.get_device_info()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "device_info": device_info
    })

@app.get("/status/{session_id}")
async def get_processing_status(session_id: str):
    """Get processing status for a session"""
    if session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    def make_json_serializable(obj):
        """Recursively convert all non-JSON-serializable objects"""
        import numpy as np
        from pathlib import Path
        
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float)):
            return obj
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {str(k): make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return make_json_serializable(obj.__dict__)
        else:
            # Last resort: convert to string
            try:
                return str(obj)
            except:
                return "<non-serializable>"
    
    # Deep convert entire status dict
    status_data = make_json_serializable(processing_status[session_id])
    
    return JSONResponse(status_data)

# Removed HTML report duplication. Use GET /api/report/{session_id} for JSON and UI renders it.

@app.get("/api/analyses")
async def list_analyses():
    """List all available analysis sessions"""
    try:
        import json
        analyses = []
        
        # Ensure reports directory exists
        REPORTS_DIR.mkdir(exist_ok=True)
        
        # Scan reports directory for JSON files
        report_files = list(REPORTS_DIR.glob("*.json"))
        logger.info(f"Found {len(report_files)} report files in {REPORTS_DIR}")
        
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    
                session_id = report_file.stem
                analyses.append({
                    "session_id": session_id,
                    "file_name": report_data.get("file_name", "Unknown"),
                    "file_type": report_data.get("file_type", "audio"),
                    "overall_score": report_data.get("overall_score", 0),
                    "voice_confidence": report_data.get("voice_confidence", 0),
                    "facial_confidence": report_data.get("facial_confidence", 0),
                    "vocabulary_score": report_data.get("vocabulary_score", 0),
                    "created_at": report_data.get("timestamp", ""),
                    "file_path": str(report_file)
                })
            except Exception as e:
                logger.warning(f"Error reading report {report_file}: {e}")
                continue
        
        # Sort by creation time (most recent first)
        analyses.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        logger.info(f"Returning {len(analyses)} analyses")
        return JSONResponse(content={"analyses": analyses})
    except Exception as e:
        logger.error(f"Error listing analyses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing analyses: {str(e)}")

@app.get("/api/report/{session_id}")
async def get_report_json(session_id: str):
    """Get report data as JSON (for API/React)"""
    try:
        # Try to load from disk first
        report = report_generator.load_report(session_id)
        
        # Fallback: Check processing_status if file not found (handles race conditions)
        if not report and session_id in processing_status:
            status_data = processing_status.get(session_id, {})
            if "report" in status_data:
                report = status_data["report"]
                logger.info(f"Report retrieved from processing_status for session: {session_id}")
        
        if not report:
            logger.warning(f"Report not found for session: {session_id} (checked disk and memory)")
            raise HTTPException(status_code=404, detail="Report not found")
        
        return JSONResponse(content=report)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report JSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting report: {str(e)}")

# Removed legacy /download endpoint in favor of /api/video/... export endpoints and /api/report for JSON.

# Video Analysis Endpoints
@app.post("/api/video/{session_id}/feedback")
async def generate_detailed_feedback(session_id: str):
    """Generate comprehensive detailed feedback for video analysis"""
    try:
        # Load report data
        report = report_generator.load_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Generate detailed feedback
        feedback = detailed_feedback_generator.generate_comprehensive_feedback(session_id, report)
        
        # Add word analysis data to feedback if available
        if 'word_analysis' in report:
            feedback['word_analysis'] = report['word_analysis']
        
        return JSONResponse(content=feedback)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating detailed feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")

class ChatRequest(BaseModel):
    message: str

class ChatStreamRequest(ChatRequest):
    pass


def run_pdf_export(job_id: str, session_id: str):
    """Background task to generate professional PDF exports."""
    try:
        export_job_manager.update_job(job_id, status="processing", message="Collecting session data")
        report = report_generator.load_report(session_id)
        if not report:
            export_job_manager.fail_job(job_id, "Report not found")
            return

        context_payload = chatbot_service.get_session_context(session_id, report)
        export_job_manager.update_job(job_id, status="rendering", message="Rendering professional PDF")
        pdf_path = pro_pdf_exporter.generate_pdf(session_id, report, context_payload)
        export_job_manager.complete_job(job_id, str(pdf_path))
    except Exception as exc:
        logger.error(f"PDF export job failed: {exc}")
        export_job_manager.fail_job(job_id, str(exc))

@app.post("/api/video/{session_id}/chat")
async def chat_with_ai(session_id: str, request: ChatRequest):
    """Chat with AI about the video analysis"""
    try:
        # Load report data
        report = report_generator.load_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Get chatbot response
        response = chatbot_service.get_response(session_id, request.message, report)
        
        return JSONResponse(content={"response": response})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/api/video/{session_id}/chat/stream")
async def chat_with_ai_stream(session_id: str, request: ChatStreamRequest):
    """Stream chatbot response chunks for richer UX."""
    try:
        report = report_generator.load_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        def event_generator():
            for chunk in chatbot_service.stream_response(session_id, request.message, report):
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Streaming chatbot error: {exc}")
        raise HTTPException(status_code=500, detail=f"Error streaming chat: {str(exc)}")

@app.get("/api/video/{session_id}/chat/context")
async def get_chat_context(session_id: str):
    """Return session insights, quick prompts, and flagged snippets for chatbot UI."""
    try:
        report = report_generator.load_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        context_payload = chatbot_service.get_session_context(session_id, report)
        return JSONResponse(content=context_payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Chat context error: {exc}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat context: {str(exc)}")

@app.post("/api/video/{session_id}/chat/reset")
async def reset_chat_context(session_id: str):
    """Clear chatbot conversation history for the session."""
    try:
        chatbot_service.clear_conversation(session_id)
        return JSONResponse(content={"status": "cleared"})
    except Exception as exc:
        logger.error(f"Chat reset error: {exc}")
        raise HTTPException(status_code=500, detail=f"Error resetting chat: {str(exc)}")

@app.post("/api/video/{session_id}/export/pdf")
async def request_pdf_export(session_id: str, background_tasks: BackgroundTasks):
    """Queue professional PDF export generation."""
    report = report_generator.load_report(session_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    job_id = export_job_manager.create_job(session_id, "pdf")
    export_job_manager.update_job(job_id, status="queued", message="Queued for rendering")
    background_tasks.add_task(run_pdf_export, job_id, session_id)
    return JSONResponse(content={"job_id": job_id, "status": "queued"})

@app.get("/api/video/export/{job_id}/status")
async def get_export_status(job_id: str):
    """Return current status for an export job."""
    job = export_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=job)

@app.get("/api/video/export/{job_id}/download")
async def download_export(job_id: str):
    """Download generated export file when ready."""
    job = export_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed" or not job.get("file_path"):
        raise HTTPException(status_code=400, detail="Export not ready")

    file_path = Path(job["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file missing")

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=file_path.name
    )

@app.get("/api/video/{session_id}/transcript")
async def get_timestamped_transcript(session_id: str):
    """Get timestamped transcript for video"""
    try:
        # Load report data
        report = report_generator.load_report(session_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # DEBUG: Log what's in the report
        logger.info(f"[TRANSCRIPT DEBUG] Report keys: {list(report.keys())}")
        logger.info(f"[TRANSCRIPT DEBUG] Has transcript: {bool(report.get('transcript'))}")
        logger.info(f"[TRANSCRIPT DEBUG] Has words_with_timing: {bool(report.get('words_with_timing'))}")
        logger.info(f"[TRANSCRIPT DEBUG] words_with_timing length: {len(report.get('words_with_timing', []))}")
        
        # Get transcript from report
        transcript = report.get('transcript', '')
        logger.info(f"[TRANSCRIPT DEBUG] Transcript length: {len(transcript)}")
        
        # VERBATIM MODE: Try to get audio results with words_with_timing
        audio_results = {}
        
        # First, try to get from processing status (if still in memory)
        if session_id in processing_status and 'audio_results' in processing_status[session_id]:
            audio_results = processing_status[session_id].get('audio_results', {})
            logger.info(f"[TRANSCRIPT DEBUG] Found audio_results in processing_status")
        
        # Second, check if words_with_timing is stored in the report (persisted)
        if not audio_results.get('words_with_timing') and report.get('words_with_timing'):
            audio_results = {
                'words_with_timing': report.get('words_with_timing', []),
                'filler_analysis': report.get('filler_analysis', {}),
                'speaking_metrics': report.get('speaking_metrics', {}),
                'transcript': transcript
            }
            logger.info(f"[TRANSCRIPT DEBUG] Reconstructed audio_results from report. words_with_timing count: {len(audio_results.get('words_with_timing', []))}")
        
        # Fallback: reconstruct from report without word timing
        if not audio_results:
            audio_results = {
                'transcript': transcript,
                'speaking_metrics': {
                    'total_duration': report.get('speaking_rate_wpm', 0) * 60 / 150 if report.get('speaking_rate_wpm') else 0
                }
            }
            logger.warning(f"[TRANSCRIPT DEBUG] Using fallback audio_results without word timing")
        
        # Generate timestamped transcript
        logger.info(f"[TRANSCRIPT DEBUG] Calling transcript_processor with audio_results keys: {list(audio_results.keys())}")
        timestamped = transcript_processor.get_timestamped_transcript(audio_results, transcript)
        logger.info(f"[TRANSCRIPT DEBUG] Generated timestamped segments: {len(timestamped)}")
        
        return JSONResponse(content={"transcript": timestamped, "full_text": transcript})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting transcript: {str(e)}")

class EnhanceRequest(BaseModel):
    transcript: Optional[str] = None

@app.post("/api/video/{session_id}/enhance-transcript")
async def enhance_transcript_endpoint(session_id: str, request: EnhanceRequest):
    """Enhance transcript using AI"""
    try:
        transcript = request.transcript or ''
        if not transcript:
            # Try to get from report
            report = report_generator.load_report(session_id)
            if report:
                transcript = report.get('transcript', '')
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        # Enhance transcript
        enhanced = transcript_enhancer.enhance_transcript(transcript)
        
        return JSONResponse(content=enhanced)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing transcript: {e}")
        raise HTTPException(status_code=500, detail=f"Error enhancing transcript: {str(e)}")

@app.post("/api/video/{session_id}/enhance-vocabulary")
async def enhance_vocabulary_endpoint(session_id: str, request: EnhanceRequest):
    """Enhance vocabulary using AI"""
    try:
        transcript = request.transcript or ''
        if not transcript:
            # Try to get from report
            report = report_generator.load_report(session_id)
            if report:
                transcript = report.get('transcript', '')
        
        if not transcript:
            raise HTTPException(status_code=400, detail="No transcript available")
        
        # Enhance vocabulary
        enhanced = vocabulary_enhancer.enhance_vocabulary(transcript)
        
        return JSONResponse(content=enhanced)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enhancing vocabulary: {e}")
        raise HTTPException(status_code=500, detail=f"Error enhancing vocabulary: {str(e)}")

@app.get("/api/video/{session_id}/file")
async def get_video_file(session_id: str):
    """Serve video/audio file for playback"""
    try:
        file_path = None
        
        # Try to get file path from processing status
        if session_id in processing_status:
            stored_path = processing_status[session_id].get('file_path')
            if stored_path:
                file_path = Path(stored_path)
                if file_path.exists():
                    pass  # Found it
                else:
                    file_path = None
        
        # If not found, try to find file in uploads directory
        if not file_path or not file_path.exists():
            upload_dir = UPLOAD_DIR
            # Search for file with session_id in name
            for file in upload_dir.glob("*"):
                if session_id in file.name and file.suffix.lower() in ALLOWED_EXTENSIONS:
                    file_path = file
                    break
        
        # Last resort: find most recent file
        if not file_path or not file_path.exists():
            upload_dir = UPLOAD_DIR
            files = sorted(upload_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            for f in files[:5]:  # Check last 5 files
                if f.suffix.lower() in ALLOWED_EXTENSIONS:
                    file_path = f
                    break
        
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        ext = file_path.suffix.lower()
        media_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac'
        }
        
        media_type = media_types.get(ext, 'application/octet-stream')
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

# Pydantic models for request validation
class RegisterRequest(BaseModel):
    username: str
    email: str  # EmailStr causes pydantic v1/v2 conflicts
    password: str
    full_name: str = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Authentication Endpoints
@app.post("/auth/register")
async def register_user(request: RegisterRequest):
    """Register a new user"""
    try:
        result = auth_manager.register_user(
            request.username, 
            request.email, 
            request.password, 
            request.full_name
        )
        return {"message": "User registered successfully", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed. Please try again.")

@app.post("/auth/login")
async def login_user(request: LoginRequest):
    """Login user"""
    try:
        result = auth_manager.authenticate_user(request.username, request.password)
        return {"message": "Login successful", "data": result}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed. Please try again.")

@app.post("/auth/logout")
async def logout_user(session_token: str):
    """Logout user"""
    success = auth_manager.logout_user(session_token)
    if success:
        return {"message": "Logout successful"}
    else:
        raise HTTPException(status_code=400, detail="Invalid session token")

# User Management Endpoints
async def get_current_user_from_header(authorization: str = Header(None)):
    """Extract user from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    token = authorization.replace("Bearer ", "")
    return auth_manager.get_current_user(token)

@app.get("/api/user/profile")
async def get_user_profile(user = Depends(get_current_user_from_header)):
    """Get user profile"""
    try:
        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_premium": user.is_premium,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve profile.")

@app.get("/api/user/analyses")
async def get_user_analyses(limit: int = 10, user = Depends(get_current_user_from_header)):
    """Get user's analysis history (JWT auth)"""
    try:
        analyses = db_manager.get_user_analyses(user.id, limit)
        return {"analyses": analyses}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Analyses error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analyses.")

@app.get("/api/sessions")
async def list_user_sessions(limit: int = 25, user = Depends(get_current_user_from_header)):
    """Return condensed session summaries for authenticated users (JWT)."""
    try:
        sessions = db_manager.get_user_sessions(user.id, limit)
        return {"sessions": sessions}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Session history error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session history.")

@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str, user = Depends(get_current_user_from_header)):
    """Fetch a single session summary."""
    try:
        summary = db_manager.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        if summary.get("user_id") and summary["user_id"] != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Session detail error: {exc}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session.")

# Backwards-compatible redirects for moved endpoints
@app.get("/user/profile")
async def redirect_user_profile():
    return RedirectResponse(url="/api/user/profile", status_code=307)

@app.get("/user/analyses")
async def redirect_user_analyses():
    return RedirectResponse(url="/api/user/analyses", status_code=307)

@app.get("/sessions")
async def redirect_sessions():
    return RedirectResponse(url="/api/sessions", status_code=307)

@app.get("/sessions/{session_id}")
async def redirect_session_details(session_id: str):
    return RedirectResponse(url=f"/api/sessions/{session_id}", status_code=307)

class FirebaseUserRequest(BaseModel):
    email: str
    uid: str
    display_name: Optional[str] = None

class FirebaseSessionsRequest(BaseModel):
    email: str
    uid: str
    display_name: Optional[str] = None
    limit: int = 50

# Support POST for analyses endpoint to match frontend expectation
@app.post("/api/firebase/analyses")
async def get_firebase_user_analyses(
    request: FirebaseUserRequest,
    limit: int = 1000
):
    """Get analysis history for Firebase-authenticated user (supports both POST and GET)"""
    try:
        # Extract user info from POST request body
        user_email = request.email
        user_uid = request.uid
        user_display_name = request.display_name
        
        session = db_manager.get_session()
        try:
            # Find or create user by email
            user = session.query(User).filter(User.email == user_email).first()
            
            if not user:
                # Create new user for Firebase auth
                # Use email as username if no conflict, otherwise use uid
                username = user_email.split('@')[0]
                existing_username = session.query(User).filter(User.username == username).first()
                if existing_username:
                    username = f"user_{user_uid[:8]}" if user_uid else f"user_{uuid.uuid4().hex[:8]}"
                
                user = User(
                    username=username,
                    email=user_email,
                    hashed_password="",  # Firebase users don't have password
                    full_name=user_display_name or username,
                    is_active=True
                )
                session.add(user)
                session.commit()
                session.refresh(user)
                logger.info(f"Created new backend user for Firebase user: {user_email}")
            
            # Get user's analyses
            analyses = db_manager.get_user_analyses(user.id, limit)
            return {"analyses": analyses}
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Firebase analyses error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analyses: {str(e)}")

async def _fetch_firebase_sessions(email: str, uid: str, display_name: Optional[str] = None, limit: int = 50):
    """Shared helper to fetch Firebase-backed session summaries."""
    try:
        logger.debug(f"_fetch_firebase_sessions called: email={email}, uid={uid[:8] if uid else None}..., limit={limit}")
        user_id = get_or_create_user(email, uid, display_name)
        if not user_id:
            logger.warning(f"No user_id found/created for email={email}, uid={uid[:8] if uid else None}...")
            return {"sessions": []}
        
        logger.info(f"Fetching sessions for user_id={user_id}, limit={limit}")
        sessions = db_manager.get_user_sessions(user_id, limit)
        logger.info(f"Found {len(sessions)} session summaries for user_id={user_id}")
        
        if not sessions:
            logger.info(f"No session summaries found, checking legacy analyses for user_id={user_id}")
            legacy = db_manager.get_user_analyses(user_id, limit)
            logger.info(f"Found {len(legacy)} legacy analyses")
            formatted = []
            for a in legacy:
                formatted.append({
                    "session_id": a.get("session_id") or a.get("id"),
                    "title": a.get("file_name") or "Untitled Session",
                    "file_name": a.get("file_name"),
                    "file_type": a.get("file_type"),
                    "overall_score": a.get("overall_score"),
                    "score_breakdown": {
                        "voice_confidence": a.get("voice_confidence"),
                        "facial_confidence": a.get("facial_confidence"),
                        "vocabulary_score": a.get("vocabulary_score"),
                    },
                    "metrics": {
                        "filler_word_count": a.get("audio_analysis", {}).get("filler_analysis", {}).get("total_fillers"),
                        "speaking_rate_wpm": a.get("audio_analysis", {}).get("transcription", {}).get("speaking_rate_wpm"),
                    },
                    "created_at": a.get("created_at"),
                })
            sessions = formatted
            logger.info(f"Converted {len(sessions)} legacy analyses to session format")
        
        return {"sessions": sessions}
    except Exception as exc:
        logger.error(f"Firebase sessions error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.post("/api/firebase/sessions")
async def get_firebase_sessions_post(request: FirebaseSessionsRequest):
    """Return session summaries for Firebase-authenticated users (POST with body)."""
    return await _fetch_firebase_sessions(
        email=request.email,
        uid=request.uid,
        display_name=request.display_name,
        limit=request.limit
    )

@app.get("/api/firebase/sessions")
async def get_firebase_sessions_get(
    email: str,
    uid: str,
    display_name: Optional[str] = None,
    limit: int = 50
):
    """Return session summaries for Firebase-authenticated users (GET query params)."""
    return await _fetch_firebase_sessions(
        email=email,
        uid=uid,
        display_name=display_name,
        limit=limit
    )

# Session management
@app.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: DeleteSessionRequest
):
    """Delete a stored analysis session and associated artifacts."""
    try:
        summary = db_manager.get_session_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        user_id = None
        if request.email:
            user_id = await asyncio.to_thread(get_or_create_user, request.email, request.uid, None)

        if summary.get("user_id") and user_id and summary["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this session")

        deleted = db_manager.delete_session(session_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found or already deleted")

        # Remove stored artifacts
        try:
            report_path = REPORTS_DIR / f"{session_id}.json"
            if report_path.exists():
                report_path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove report file for %s: %s", session_id, exc)

        try:
            pdf_path = summary.get("pdf_path")
            if pdf_path:
                pdf_file = Path(pdf_path)
                if not pdf_file.is_absolute():
                    pdf_file = REPORTS_DIR / pdf_file
                if pdf_file.exists():
                    pdf_file.unlink()
        except Exception as exc:
            logger.warning("Failed to remove PDF for %s: %s", session_id, exc)

        try:
            highlights = summary.get("highlights") or {}
            thumbnail_url = highlights.get("thumbnail_url")
            if thumbnail_url:
                thumbnail_name = Path(thumbnail_url).name
                thumb_path = THUMBNAIL_DIR / thumbnail_name
                if thumb_path.exists():
                    thumb_path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove thumbnail for %s: %s", session_id, exc)

        # Remove professional export if it exists
        try:
            export_path = REPORTS_DIR / "exports" / f"{session_id}_professional.pdf"
            if export_path.exists():
                export_path.unlink()
        except Exception as exc:
            logger.warning("Failed to remove export PDF for %s: %s", session_id, exc)

        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Delete session error for %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(exc)}")


@app.delete("/api/sessions")
async def delete_all_sessions(request: DeleteSessionRequest):
    """Delete all stored analysis sessions for the authenticated user."""
    if not request.email:
        raise HTTPException(status_code=400, detail="Email is required to delete history.")

    try:
        user_id = await asyncio.to_thread(get_or_create_user, request.email, request.uid, None)
        if not user_id:
            raise HTTPException(status_code=404, detail="User not found")

        # Capture existing session metadata before deletion for cleanup.
        existing_sessions = await asyncio.to_thread(db_manager.get_user_sessions, user_id, 1000)
        existing_lookup = {session.get("session_id"): session for session in existing_sessions if session.get("session_id")}

        deleted_ids = await asyncio.to_thread(db_manager.delete_all_sessions, user_id)

        exports_dir = REPORTS_DIR / "exports"

        for session_id in deleted_ids:
            session_meta = existing_lookup.get(session_id, {})

            processing_status.pop(session_id, None)

            # Remove stored report JSON
            try:
                report_path = REPORTS_DIR / f"{session_id}.json"
                if report_path.exists():
                    report_path.unlink()
            except Exception as exc:
                logger.warning("Failed to remove report file for %s: %s", session_id, exc)

            # Remove stored PDF if available
            try:
                pdf_path = session_meta.get("pdf_path")
                if pdf_path:
                    pdf_file = Path(pdf_path)
                    if not pdf_file.is_absolute():
                        pdf_file = REPORTS_DIR / pdf_file
                    if pdf_file.exists():
                        pdf_file.unlink()
            except Exception as exc:
                logger.warning("Failed to remove PDF for %s: %s", session_id, exc)

            # Remove thumbnails
            try:
                highlights = session_meta.get("highlights") or {}
                thumbnail_url = highlights.get("thumbnail_url")
                if thumbnail_url:
                    thumbnail_name = Path(thumbnail_url).name
                    thumb_path = THUMBNAIL_DIR / thumbnail_name
                    if thumb_path.exists():
                        thumb_path.unlink()
            except Exception as exc:
                logger.warning("Failed to remove thumbnail for %s: %s", session_id, exc)

            # Remove professional export if present
            try:
                export_path = exports_dir / f"{session_id}_professional.pdf"
                if export_path.exists():
                    export_path.unlink()
            except Exception as exc:
                logger.warning("Failed to remove export PDF for %s: %s", session_id, exc)

        return {
            "status": "deleted",
            "deleted_sessions": deleted_ids,
            "count": len(deleted_ids)
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Delete all sessions error for %s: %s", request.email, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(exc)}")

# Enhanced Analysis Endpoints
class EnhancedAnalysisRequest(BaseModel):
    session_id: str

# Removed internal enhanced analysis endpoint to avoid duplicate analysis entrypoints.

@app.get("/analytics/dashboard")
async def get_analytics_dashboard(user = Depends(get_current_user_from_header)):
    """Get analytics dashboard data"""
    try:
        analyses = db_manager.get_user_analyses(user.id, limit=50)
        
        # Calculate analytics
        scores = [analysis['overall_score'] for analysis in analyses if analysis['overall_score']]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Progress over time
        progress_data = []
        for i, analysis in enumerate(analyses[:10]):  # Last 10 analyses
            progress_data.append({
                "date": analysis['created_at'],
                "score": analysis['overall_score'],
                "file_type": analysis['file_type']
            })
        
        return {
            "total_analyses": len(analyses),
            "average_score": round(avg_score, 2),
            "progress_data": progress_data,
            "file_types": {
                "audio": len([a for a in analyses if a['file_type'] == 'audio']),
                "video": len([a for a in analyses if a['file_type'] == 'video'])
            }
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analytics.")

@app.get("/cors-test")
async def cors_test():
    """Test CORS configuration"""
    return JSONResponse({"message": "CORS is working!", "origin": "http://localhost:8005"})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    device_info = device_manager.get_device_info()
    
    # List all registered routes for debugging
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else []
            })
    
    return JSONResponse({
        "status": "healthy",
        "device_info": device_info,
        "models_loaded": {
            "whisper": audio_analyzer.whisper_model is not None,
            "spacy": text_analyzer.spacy_model is not None,
            "facial": facial_analyzer.face_detector is not None,
            "openai": openai_enhancer.enabled,
            "database": True
        },
        "features": {
            "offline_mode": True,
            "openai_api": USE_OPENAI_API,  # OpenAI free tier only
            "google_cloud": False,  # Disabled - using local Whisper
            "azure_cloud": False,  # Disabled - using local DeepFace
            "user_auth": True,
            "real_time": True,
            "hybrid_mode": True
        },
        "registered_routes": routes
    })

@app.get("/routes")
async def list_routes():
    """List all registered routes for debugging"""
    routes = []
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            routes.append({
                "path": route.path,
                "methods": list(route.methods) if route.methods else [],
                "name": getattr(route, "name", "N/A")
            })
    return JSONResponse({"routes": routes})

if __name__ == "__main__":
    # Print startup information
    device_info = device_manager.get_device_info()
    logger.info("=" * 50)
    logger.info("Face2Phase - Hybrid AI Communication Feedback System")
    logger.info("=" * 50)
    logger.info(f"Device: {device_info['device']}")
    logger.info(f"GPU Available: {device_info['is_gpu']}")
    if device_info['is_gpu']:
        logger.info(f"GPU: {device_info['gpu_name']}")
    logger.info(f"spaCy Model: {'Loaded' if device_info['spacy_model_loaded'] else 'Not loaded'}")
    logger.info("=" * 50)
    
    # Start the server
    import os
    # Default to port 8000 to match frontend configuration
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
