"""
Enhanced Audio Analysis with Precise Gap Detection, Pitch Tracking, and Filler Detection
"""
# CRITICAL FIX: Disable numba JIT BEFORE importing librosa to fix compatibility issues
import os
# Disable numba JIT compilation completely to avoid compatibility issues
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_DISABLE_CUDA'] = '1'
# Force numba to use object mode (interpreter mode) instead of nopython mode
try:
    import numba
    numba.config.DISABLE_JIT = True
    # Disable numba's automatic JIT compilation
    if hasattr(numba, 'jit'):
        # Create a no-op decorator
        original_jit = numba.jit
        def noop_jit(*args, **kwargs):
            def decorator(func):
                return func
            if args and callable(args[0]):
                return args[0]
            return decorator
        numba.jit = noop_jit
except ImportError:
    pass  # Numba not installed, that's fine

# Attempt to import Whisper; if unavailable, set to None
try:
    import whisper
except ImportError:
    whisper = None  # Whisper optional; functionality will be limited
import librosa
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import soundfile as sf
from ...core.settings import (
    FILLER_WORDS, MUMBLING_FILLERS, LONG_PAUSE_THRESHOLD, SHORT_PAUSE_THRESHOLD, 
    MEDIUM_PAUSE_THRESHOLD, WHISPER_MODEL_SIZE, WHISPER_LANGUAGE, WHISPER_TEMPERATURE,
    WHISPER_NO_SPEECH_THRESHOLD, WHISPER_LOGPROB_THRESHOLD, WHISPER_COMPRESSION_RATIO_THRESHOLD,
    MIN_TRANSCRIPTION_CONFIDENCE, VALIDATION_ENABLED, DISABLE_ACOUSTIC_FILLER_DETECTION,
    ACOUSTIC_FILLER_MIN_CONFIDENCE, ACOUSTIC_FILLER_MIN_RMS,
    WPM_SLOW_THRESHOLD, WPM_FAST_THRESHOLD, WPM_MIN_PLAUSIBLE, WPM_MAX_PLAUSIBLE
)

from ..speech.disfluency_detector import disfluency_detector
from ..text.sentence_opener_analyzer import analyze_sentence_openers
from ...utils.device_detector import device_manager
from ...utils.report_utils import (
    compute_filler_trend,
    compute_pause_cadence,
    compute_opening_confidence
)

logger = logging.getLogger(__name__)

# Voice Activity Detection (VAD)
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    webrtcvad = None
    VAD_AVAILABLE = False
    logger.warning("webrtcvad not available. Install with: pip install webrtcvad")

class EnhancedAudioAnalyzer:
    """Enhanced audio analyzer with precise gap detection and pitch tracking"""
    
    def __init__(self):
        self.device = device_manager.get_device()
        self.whisper_model = None
        self._whisper_loaded = False  # Track if we've attempted to load
        self.vad = None
        self._setup_ffmpeg()
        # LAZY LOADING: Don't load Whisper at startup
        # self._load_whisper_model()  # Commented out for fast startup
        self._initialize_vad()
    
    def _setup_ffmpeg(self):
        """Set up FFmpeg path for Windows compatibility"""
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            import os
            os.environ["FFMPEG_BINARY"] = ffmpeg_path
            logger.info(f"Set FFmpeg path: {ffmpeg_path}")
        except Exception as e:
            logger.warning(f"Could not set FFmpeg path: {e}")
    
    def _ensure_whisper_loaded(self):
        """Lazy load Whisper model on first use"""
        if self._whisper_loaded:
            return self.whisper_model is not None
        
        self._whisper_loaded = True
        return self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model if available"""
        if whisper is None:
            logger.error("❌ CRITICAL: Whisper library not installed; audio transcription will be disabled.")
            logger.error("   Install with: pip install openai-whisper")
            self.whisper_model = None
            return False
        try:
            logger.info(f"🔄 Loading Whisper model: {WHISPER_MODEL_SIZE} on device: {self.device}")
            logger.info(f"   Model size: {WHISPER_MODEL_SIZE}")
            logger.info(f"   Device: {self.device}")
            
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=self.device)
            
            logger.info(f"✅ Successfully loaded Whisper model: {WHISPER_MODEL_SIZE}")
            logger.info(f"   Whisper is ready for transcription!")
            return True
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR loading Whisper model: {e}", exc_info=True)
            logger.error(f"   Model size attempted: {WHISPER_MODEL_SIZE}")
            logger.error(f"   Device attempted: {self.device}")
            logger.error("   This will cause all transcriptions to fail!")
            self.whisper_model = None
            return False
    
    def _initialize_vad(self):
        """Initialize Voice Activity Detection"""
        if VAD_AVAILABLE:
            try:
                # Use mode 2 (aggressive) for better speech detection
                self.vad = webrtcvad.Vad(2)
                logger.info("Voice Activity Detection (VAD) initialized")
            except Exception as e:
                logger.warning(f"Could not initialize VAD: {e}")
                self.vad = None
        else:
            self.vad = None
            logger.info("VAD not available (webrtcvad not installed)")

    def _load_audio_safe(self, audio_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio safely bypassing librosa.load's numba-dependent resampling.
        Ensures output is MONO and resampled to target_sr using scipy.
        CRITICAL FIX for:
        1. Numba/Resampy compatibility issues
        2. Stereo->Mono conversion for Whisper (fixes 1.9TB memory crash)
        """
        try:
            import soundfile as sf
            from scipy import signal
            
            # 1. Read with soundfile (no numba)
            # soundfile.read returns (samples, channels) for stereo
            audio_data, sr_native = sf.read(str(audio_path))
            
            # 2. Convert to Mono if stereo/multichannel
            if audio_data.ndim > 1:
                # Average across channels
                audio_data = np.mean(audio_data, axis=1)
                
            # 3. Resample if needed
            if sr_native != target_sr:
                num_samples = int(len(audio_data) * target_sr / sr_native)
                # Cast to float32 before resampling to save memory
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data = signal.resample(audio_data, num_samples)
                sr = target_sr
            else:
                sr = sr_native
                
            # Ensure float32 (scipy/soundfile might return float64)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            return audio_data, sr
            
        except Exception as e:
            logger.warning(f"Safe audio load failed: {e}, falling back to librosa")
            # Fallback (might crash if numba issue persists)
            return librosa.load(str(audio_path), sr=target_sr)

    def _load_audio_safe(self, audio_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Load audio safely bypassing librosa.load's numba-dependent resampling.
        Ensures output is MONO and resampled to target_sr using scipy.
        CRITICAL FIX for:
        1. Numba/Resampy compatibility issues
        2. Stereo->Mono conversion for Whisper (fixes 1.9TB memory crash)
        """
        try:
            import soundfile as sf
            from scipy import signal
            
            # 1. Read with soundfile (no numba)
            # soundfile.read returns (samples, channels) for stereo
            audio_data, sr_native = sf.read(str(audio_path))
            
            # 2. Convert to Mono if stereo/multichannel
            if audio_data.ndim > 1:
                # Average across channels
                audio_data = np.mean(audio_data, axis=1)
                
            # 3. Resample if needed
            if sr_native != target_sr:
                num_samples = int(len(audio_data) * target_sr / sr_native)
                # Cast to float32 before resampling to save memory
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                audio_data = signal.resample(audio_data, num_samples)
                sr = target_sr
            else:
                sr = sr_native
                
            # Ensure float32 (scipy/soundfile might return float64)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            return audio_data, sr
            
        except Exception as e:
            logger.warning(f"Safe audio load failed: {e}, falling back to librosa")
            # Fallback (might crash if numba issue persists)
            return librosa.load(str(audio_path), sr=target_sr)
    
    def calculate_adaptive_threshold(self, audio_data: np.ndarray, sr: int) -> float:
        """
        Calculate adaptive energy threshold based on audio characteristics
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Adaptive threshold value
        """
        try:
            # Calculate frame-level energy (RMS)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Calculate baseline noise floor (bottom 10% as noise)
            noise_floor = np.percentile(rms, 10)
            
            # Calculate speech energy distribution (median as typical speech)
            speech_energy = np.percentile(rms, 50)
            
            # Adaptive threshold: 30% above noise floor, but not more than 50% of speech energy
            threshold = noise_floor + (speech_energy - noise_floor) * 0.3
            
            # Ensure minimum threshold to avoid false positives
            min_threshold = 0.01
            max_threshold = speech_energy * 0.5
            
            # Clamp threshold
            threshold = max(min_threshold, min(threshold, max_threshold))
            
            logger.info(f"Adaptive threshold calculated: {threshold:.4f} (noise_floor: {noise_floor:.4f}, speech: {speech_energy:.4f})")
            
            return float(threshold)
        except Exception as e:
            logger.warning(f"Error calculating adaptive threshold: {e}, using default")
            return 0.015
    
    def detect_precise_pauses(self, audio_path: Path, energy_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect precise pauses in audio using adaptive energy-based detection
        
        Args:
            audio_path: Path to audio file
            energy_threshold: Optional fixed threshold (if None, uses adaptive)
            
        Returns:
            List of pause dictionaries with start, end, duration
        """
        try:
            # Load audio SAFE
            y, sr = self._load_audio_safe(audio_path, target_sr=16000)
            
            # Calculate adaptive threshold if not provided
            if energy_threshold is None:
                energy_threshold = self.calculate_adaptive_threshold(y, sr)
                logger.info(f"Using adaptive energy threshold: {energy_threshold:.4f}")
            
            # Calculate frame-level energy (RMS)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Convert frames to time
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            # Use VAD if available for more accurate speech/silence detection
            if self.vad is not None:
                # VAD works on 10ms, 20ms, or 30ms frames at specific sample rates
                # Convert to 16kHz, 16-bit PCM for VAD
                try:
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                        sr = 16000
                    
                    # Convert to int16 PCM
                    y_int16 = (y * 32767).astype(np.int16)
                    
                    # VAD frame size: 30ms at 16kHz = 480 samples
                    frame_duration_ms = 30
                    frame_size = int(sr * frame_duration_ms / 1000)
                    
                    # Detect speech using VAD
                    is_silent_vad = []
                    for i in range(0, len(y_int16), frame_size):
                        frame = y_int16[i:i+frame_size]
                        if len(frame) < frame_size:
                            # Pad last frame
                            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
                        
                        try:
                            is_speech = self.vad.is_speech(frame.tobytes(), sr)
                            is_silent_vad.append(not is_speech)
                        except:
                            # Fallback to energy-based
                            frame_energy = np.mean(np.abs(frame.astype(np.float32) / 32767))
                            is_silent_vad.append(frame_energy < energy_threshold)
                    
                    # Upsample VAD results to match RMS frame rate
                    rms_frames = len(rms)
                    vad_frames = len(is_silent_vad)
                    if vad_frames > 0:
                        # Interpolate VAD results to match RMS frame count
                        vad_indices = np.linspace(0, vad_frames - 1, rms_frames).astype(int)
                        is_silent = np.array([is_silent_vad[i] for i in vad_indices])
                        logger.info("Using VAD for speech/silence detection")
                    else:
                        is_silent = rms < energy_threshold
                except Exception as e:
                    logger.warning(f"VAD processing failed: {e}, using energy-based detection")
                    is_silent = rms < energy_threshold
            else:
                # Fallback to energy-based detection
                is_silent = rms < energy_threshold
            
            # Find continuous silent regions
            pauses = []
            in_pause = False
            pause_start = 0
            
            for i, silent in enumerate(is_silent):
                if silent and not in_pause:
                    # Start of pause
                    in_pause = True
                    pause_start = times[i]
                elif not silent and in_pause:
                    # End of pause
                    pause_end = times[i]
                    duration = pause_end - pause_start
                    
                    # Only record significant pauses (>= 0.3 seconds)
                    if duration >= 0.3:
                        pauses.append({
                            "start": float(pause_start),
                            "end": float(pause_end),
                            "duration": float(duration),
                            "type": "LONG_PAUSE" if duration >= LONG_PAUSE_THRESHOLD else "SHORT_PAUSE"
                        })
                    in_pause = False
            
            # Handle case where audio ends in silence
            if in_pause:
                pause_end = times[-1]
                duration = pause_end - pause_start
                if duration >= 0.3:
                    pauses.append({
                        "start": float(pause_start),
                        "end": float(pause_end),
                        "duration": float(duration),
                        "type": "LONG_PAUSE" if duration >= LONG_PAUSE_THRESHOLD else "SHORT_PAUSE"
                    })
            
            logger.info(f"Detected {len(pauses)} pauses in audio")
            if pauses:
                logger.info(f"Sample pause durations: {[p.get('duration', 0) for p in pauses[:3]]}")
            else:
                logger.warning("No pauses detected from audio - may need fallback to Whisper segments")
            return pauses if pauses else []
            
        except Exception as e:
            logger.error(f"Error detecting pauses: {e}")
            return []
    
    def _extract_pauses_from_whisper_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract pauses from gaps between Whisper transcription segments.
        This is a fallback when audio-based pause detection fails.
        
        Args:
            segments: List of Whisper segments with 'start' and 'end' times
            
        Returns:
            List of pause dictionaries
        """
        pauses = []
        if not segments or len(segments) < 2:
            return pauses
        
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            current_end = current_segment.get("end", 0)
            next_start = next_segment.get("start", 0)
            
            # Calculate gap between segments
            gap_duration = next_start - current_end
            
            # Only record gaps >= 0.3 seconds (same threshold as audio detection)
            if gap_duration >= 0.3:
                pauses.append({
                    "start": float(current_end),
                    "end": float(next_start),
                    "duration": float(gap_duration),
                    "type": "LONG_PAUSE" if gap_duration >= LONG_PAUSE_THRESHOLD else "SHORT_PAUSE",
                    "source": "whisper_segments"  # Mark as fallback source
                })
        
        logger.info(f"Extracted {len(pauses)} pauses from {len(segments)} Whisper segments")
        return pauses
    
    def _extract_pauses_from_word_timing(self, words_with_timing: List[Dict]) -> List[Dict]:
        """
        Extract pauses from gaps between words in Whisper transcription.
        This is the most reliable method as it uses actual word timestamps.
        
        Args:
            words_with_timing: List of words with 'start' and 'end' times
            
        Returns:
            List of pause dictionaries
        """
        pauses = []
        if not words_with_timing or len(words_with_timing) < 2:
            return pauses
        
        for i in range(len(words_with_timing) - 1):
            current_word = words_with_timing[i]
            next_word = words_with_timing[i + 1]
            
            current_end = current_word.get("end", 0)
            next_start = next_word.get("start", 0)
            
            # Calculate gap between words
            gap_duration = next_start - current_end
            
            # Only record gaps >= 0.3 seconds (same threshold as audio detection)
            if gap_duration >= 0.3:
                pauses.append({
                    "start": float(current_end),
                    "end": float(next_start),
                    "duration": float(gap_duration),
                    "type": "LONG_PAUSE" if gap_duration >= LONG_PAUSE_THRESHOLD else "SHORT_PAUSE",
                    "source": "word_timing"  # Mark as fallback source
                })
        
        logger.info(f"Extracted {len(pauses)} pauses from {len(words_with_timing)} word timestamps")
        if pauses:
            logger.info(f"Pause durations: {[p.get('duration', 0) for p in pauses[:5]]}")
        return pauses
    
    def track_pitch_variations(self, audio_path: Path) -> Dict:
        """
        Track pitch variations to detect high/low voice changes
        
        Uses fallback methods if librosa.pyin fails due to numba compatibility issues.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with pitch tracking data
        """
        try:
            # Load audio SAFE
            y, sr = self._load_audio_safe(audio_path, target_sr=16000)
            
            # Try librosa.pyin first (may fail with numba 0.57.1)
            f0 = None
            voiced_flag = None
            try:
                # Attempt pyin with numba disabled
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, 
                    fmin=librosa.note_to_hz('C2'), 
                    fmax=librosa.note_to_hz('C7'),
                    frame_length=2048
                )
            except Exception as pyin_error:
                # Fallback to piptrack if pyin fails (numba issue)
                logger.warning(f"librosa.pyin failed (likely numba issue), using fallback method: {pyin_error}")
                try:
                    # Use piptrack as fallback (simpler, less numba-dependent)
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                    
                    # Extract pitch values frame by frame
                    f0_values = []
                    times_list = []
                    hop_length = 512
                    
                    for t in range(pitches.shape[1]):
                        # Find strongest pitch in this frame
                        index = magnitudes[:, t].argmax()
                        pitch_val = pitches[index, t]
                        
                        if pitch_val > 0:  # Valid pitch
                            f0_values.append(pitch_val)
                            times_list.append(t * hop_length / sr)
                    
                    if len(f0_values) > 0:
                        # Convert to numpy arrays similar to pyin output
                        f0 = np.array(f0_values)
                        voiced_flag = np.ones(len(f0_values), dtype=bool)
                        logger.info(f"Used piptrack fallback: extracted {len(f0_values)} pitch points")
                    else:
                        logger.warning("Both pyin and piptrack failed to extract pitch, using energy-based estimation")
                        # Last resort: estimate from spectral centroid (very basic)
                        f0 = self._estimate_pitch_from_energy(y, sr)
                        voiced_flag = np.ones(len(f0), dtype=bool) if len(f0) > 0 else np.array([], dtype=bool)
                except Exception as fallback_error:
                    logger.warning(f"Piptrack fallback also failed: {fallback_error}, using basic estimation")
                    # Last resort: very basic pitch estimation
                    f0 = self._estimate_pitch_from_energy(y, sr)
                    voiced_flag = np.ones(len(f0), dtype=bool) if len(f0) > 0 else np.array([], dtype=bool)
            
            # Handle case where we still have no pitch data
            if f0 is None or len(f0) == 0:
                logger.warning("No pitch data available, returning empty pitch analysis")
                return {
                    "pitch_variations": [],
                    "avg_pitch": 0.0,
                    "pitch_range": 0.0,
                    "pitch_std": 0.0,
                    "pitch_timeline": []
                }
            
            # Convert frames to time if we used pyin
            if voiced_flag is not None and len(voiced_flag) == len(f0):
                # We have voiced_flag from pyin
                times = librosa.times_like(f0, sr=sr) if len(f0) > 0 else np.array([])
                voiced_f0 = f0[voiced_flag]
                voiced_times = times[voiced_flag] if len(times) == len(f0) else np.array([])
            else:
                # We used fallback method
                if len(f0) > 0 and len(times_list) == len(f0):
                    voiced_f0 = f0
                    voiced_times = np.array(times_list)
                else:
                    # Generate times for pitch data
                    hop_length = 512
                    times = librosa.times_like(f0, sr=sr, hop_length=hop_length) if len(f0) > 0 else np.array([])
                    voiced_f0 = f0
                    voiced_times = times
            
            if len(voiced_f0) == 0:
                return {
                    "pitch_variations": [],
                    "avg_pitch": 0.0,
                    "pitch_range": 0.0,
                    "pitch_std": 0.0,
                    "pitch_timeline": []
                }
            
            # Calculate statistics
            avg_pitch = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            std_pitch = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            
            # Classify pitch levels
            pitch_variations = []
            pitch_timeline = []
            
            for i, (time, pitch) in enumerate(zip(voiced_times, voiced_f0)):
                if not np.isnan(pitch) and pitch > 0:
                    # Classify as high, normal, or low
                    if avg_pitch > 0:
                        if pitch > avg_pitch + std_pitch:
                            level = "HIGH"
                        elif pitch < avg_pitch - std_pitch:
                            level = "LOW"
                        else:
                            level = "NORMAL"
                    else:
                        level = "NORMAL"
                    
                    pitch_variations.append({
                        "time": float(time),
                        "pitch": float(pitch),
                        "level": level
                    })
                    
                    pitch_timeline.append({
                        "time": float(time),
                        "pitch": float(pitch)
                    })
            
            logger.info(f"Tracked {len(pitch_variations)} pitch points (avg: {avg_pitch:.1f} Hz, std: {std_pitch:.1f} Hz)")
            
            pitch_range = float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)) if len(voiced_f0) > 0 else 0.0
            
            return {
                "pitch_variations": pitch_variations,
                "avg_pitch": avg_pitch,
                "pitch_range": pitch_range,
                "pitch_std": std_pitch,
                "pitch_timeline": pitch_timeline
            }
            
        except Exception as e:
            logger.error(f"Error tracking pitch: {e}", exc_info=True)
            return {
                "pitch_variations": [],
                "avg_pitch": 0.0,
                "pitch_range": 0.0,
                "pitch_std": 0.0,
                "pitch_timeline": [],
                "error": str(e)
            }
    
    def _estimate_pitch_from_energy(self, y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
        """
        Basic pitch estimation using spectral centroid as fallback
        This is a simple method that doesn't rely on numba
        """
        try:
            # Use spectral centroid as rough pitch indicator
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
            
            # Filter out very low/high values (likely noise)
            valid_mask = (spectral_centroids > 80) & (spectral_centroids < 400)
            valid_centroids = spectral_centroids[valid_mask]
            
            if len(valid_centroids) > 0:
                # Convert spectral centroid to approximate F0 (rough approximation)
                # This is not perfect but gives some pitch data
                f0_estimate = valid_centroids * 0.5  # Rough scaling factor
                return f0_estimate
            else:
                return np.array([])
        except Exception as e:
            logger.warning(f"Basic pitch estimation failed: {e}")
            return np.array([])
    
    def transcribe_with_word_timestamps(self, audio_path: Path) -> Dict:
        """
        Transcribe audio with word-level timestamps using Whisper - VERBATIM MODE
        
        Returns raw Whisper output with ALL words including filler sounds (uh, um, ah).
        No filtering or post-processing applied.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detailed transcription with word-level timing
        """
        try:
            # LAZY LOADING: Load Whisper model on first use
            if not self.whisper_model:
                self._ensure_whisper_loaded()
            
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            logger.info(f"Starting VERBATIM transcription: {audio_path}")
            
            # CRITICAL FIX: Use safe loader to avoid Numba errors and Memory crashes
            logger.info("Loading audio with SAFE LOADER (soundfile + scipy + mono conversion)...")
            audio_data, sr = self._load_audio_safe(audio_path, target_sr=16000)
            
            audio_duration = len(audio_data) / sr
            logger.info(f"✅ Audio ready: {len(audio_data)} samples, {sr} Hz, {audio_duration:.1f}s duration")
            
            # VERBATIM TRANSCRIPTION SETTINGS for clean audio with natural fillers
            logger.info("Transcribing with VERBATIM settings (preserving natural speech)...")
            
            logger.info("🎤 Starting Whisper transcription now...")
            logger.info(f"   This may take a while depending on audio length ({audio_duration:.1f}s)")
            
            result = self.whisper_model.transcribe(
                audio_data,
                word_timestamps=True,  # CRITICAL: Enable word timestamps for pause detection
                verbose=False,
                # Core settings
                language=WHISPER_LANGUAGE,
                task='transcribe',  # Not 'translate' - preserve original
                # Verbatim settings
                temperature=0.0,  # Deterministic
                no_speech_threshold=0.4,  # Standard threshold
                logprob_threshold=None,  # Keep all words
                compression_ratio_threshold=2.4,  # Standard
                condition_on_previous_text=True,
                fp16=False,
                # CRITICAL: Prompt that encourages keeping natural speech patterns
                initial_prompt=(
                    "This is a natural speech recording. "
                    "Transcribe exactly as spoken, including any natural hesitations, "
                    "filler words like um, uh, ah, er, oh, and repetitions. "
                    "Keep all words verbatim."
                )
            )
            
            logger.info(f"✅ Whisper transcription completed!")
            
            # Get raw transcript - this is Whisper's direct output
            transcript = result["text"].strip()
            segments = result.get("segments", [])
            
            logger.info(f"Raw transcription: {len(transcript)} characters, {len(segments)} segments")
            logger.info(f"Transcript preview: {transcript[:200]}..." if len(transcript) > 200 else f"Transcript: {transcript}")
            
            # Extract ALL words from segments - NO FILTERING
            words_with_timing = []
            
            for segment in segments:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word = word_info.get("word", "").strip()
                        if word:  # Only skip completely empty strings
                            words_with_timing.append({
                                "word": word,
                                "start": word_info.get("start", 0),
                                "end": word_info.get("end", 0),
                                "probability": word_info.get("probability", 1.0)
                            })
            
            # Log filler word detection
            filler_words_found = [w["word"].lower().strip() for w in words_with_timing 
                                  if w["word"].lower().strip() in {"um", "uh", "ah", "er", "hmm", "erm", "uhh", "umm", "ahh"}]
            if filler_words_found:
                logger.info(f"Filler words detected: {filler_words_found}")
            else:
                logger.info("No explicit filler words detected in Whisper output")
            
            logger.info(f"Extracted {len(words_with_timing)} words with timestamps")
            
            return {
                "transcript": transcript,
                "segments": segments,
                "words_with_timing": words_with_timing,
                "duration": audio_duration,
                "transcription_quality": {
                    "avg_confidence": 1.0,
                    "avg_compression": 0.0,
                    "filtered_words": 0,
                    "is_high_quality": True,
                    "mode": "verbatim"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Error transcribing audio: {e}", exc_info=True)
            logger.error(f"   Audio path: {audio_path}")
            logger.error(f"   Whisper model loaded: {self.whisper_model is not None}")
            logger.error("   Returning empty transcript - THIS IS WHY TRANSCRIPT IS EMPTY!")
            return {
                "transcript": "",
                "segments": [],
                "words_with_timing": [],
                "duration": 0,
                "transcription_quality": {
                    "avg_confidence": 0.0,
                    "avg_compression": 0.0,
                    "filtered_words": 0,
                    "is_high_quality": False,
                    "error": str(e)
                }
            }
    
    def create_detailed_transcript(
        self, 
        words_with_timing: List[Dict], 
        pauses: List[Dict],
        pitch_data: Dict,
        audio_path: Path
    ) -> str:
        """
        Create ultra-detailed transcript with gaps, pitch, filler detection, and ADVANCED mumbling detection
        
        Args:
            words_with_timing: Word-level timing from Whisper
            pauses: Detected pauses from audio analysis
            pitch_data: Pitch tracking data
            audio_path: Path to audio file
            
        Returns:
            Detailed enhanced transcript with mumbling annotations
        """
        if not words_with_timing:
            return ""
        
        transcript_lines = []
        filler_words_list = ["um", "uh", "ah", "eh", "er", "hmm", "erm", "umm", "uhh", "ahh", "uhhh", "ummm", "ahhh", "ehhh"]
        
        # Get mumbling instances from filler analysis
        transcript_text = " ".join([w.get("word", "") for w in words_with_timing])
        estimated_duration = 0.0
        if words_with_timing:
            last_entry = words_with_timing[-1]
            estimated_duration = float(last_entry.get("end") or last_entry.get("start") or 0.0)
        filler_analysis = self._analyze_fillers(
            transcript_text,
            words_with_timing=words_with_timing,
            audio_path=audio_path,
            total_duration=estimated_duration
        )
        mumbling_instances = {inst.get('timestamp', 0): inst for inst in filler_analysis.get('mumbling_instances', []) if inst.get('timestamp')}
        stammering_instances = {inst.get('timestamp', 0): inst for inst in filler_analysis.get('stammering_instances', []) if inst.get('timestamp')}
        
        # Load audio for voice quality analysis
        try:
            audio_data, sr = self._load_audio_safe(audio_path, target_sr=16000)
        except:
            audio_data = None
            sr = 16000
        
        current_line = ""
        current_time = 0
        last_end_time = 0
        
        for i, word_info in enumerate(words_with_timing):
            word = word_info["word"]
            start = word_info["start"]
            end = word_info["end"]
            
            # Detect gap before this word
            gap_duration = start - last_end_time
            
            # Check if this is a filler word
            clean_word = word.lower().strip('.,!?;:')
            is_filler = clean_word in filler_words_list
            
            # ADVANCED: Check for mumbling at this timestamp (within 0.2s tolerance)
            is_mumbling = False
            mumbling_info = None
            for mumble_time, mumble_data in mumbling_instances.items():
                if abs(mumble_time - start) < 0.2:
                    is_mumbling = True
                    mumbling_info = mumble_data
                    break
            
            # Check for stammering at this timestamp
            is_stammering = False
            stammering_info = None
            for stammer_time, stammer_data in stammering_instances.items():
                if abs(stammer_time - start) < 0.2:
                    is_stammering = True
                    stammering_info = stammer_data
                    break
            
            # Get pitch at this moment
            pitch_level = self._get_pitch_at_time(pitch_data, start)
            
            # Get voice energy at this moment
            voice_quality = self._get_voice_quality_at_time(audio_data, sr, start, end)
            
            # ADVANCED: Detect unclear/mumbled audio segments using acoustic features
            is_unclear = False
            if audio_data is not None:
                try:
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    if start_sample < len(audio_data) and end_sample <= len(audio_data):
                        segment = audio_data[start_sample:end_sample]
                        # Use acoustic features to detect mumbling
                        if len(segment) > 0:
                            zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
                            rms = np.mean(librosa.feature.rms(y=segment)[0])
                            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)[0])
                            
                            # Mumbling characteristics: VERY strict criteria
                            # high ZCR (noisy), VERY low RMS (quiet), low spectral centroid
                            # Only flag if audio is actually quiet AND has mumbling characteristics
                            if zcr > 0.15 and rms < 0.01 and rms > 0.001 and spectral_centroid < 1500:
                                is_unclear = True
                except:
                    pass
            
            # Start new line if significant gap or new sentence
            if gap_duration >= 0.5 or i == 0:
                # Save previous line
                if current_line:
                    transcript_lines.append(current_line)
                
                # Add gap marker if significant
                gap_marker = ""
                if gap_duration >= 5.0:
                    gap_marker = f" <{gap_duration:.1f}sec GAP DETECTED>"
                elif gap_duration >= 2.0:
                    gap_marker = f" <{gap_duration:.1f}sec pause>"
                elif gap_duration >= 0.5:
                    gap_marker = f" <{gap_duration:.1f}s>"
                
                # Start new line with timestamp in MM:SS format (like Riverside.fm)
                minutes = int(start // 60)
                seconds = int(start % 60)
                timestamp_formatted = f"{minutes:02d}:{seconds:02d}"
                current_line = f"[{timestamp_formatted}]{gap_marker} "
                current_time = start
            
            # Add word with annotations
            word_annotation = word
            
            # PRIORITY: Add mumbling annotation (highest priority)
            if is_mumbling and mumbling_info:
                mumble_word = mumbling_info.get('word', word)
                # Show mumbling in brackets with visual indicator
                word_annotation = f"[{mumble_word}]"  # e.g., [uhhh], [umm]
            # Add stammering annotation
            elif is_stammering and stammering_info:
                stammer_type = stammering_info.get('type', 'repetition')
                if stammer_type == 'word_repetition':
                    word_annotation = f"{word}-{word}..."  # Show repetition
                elif stammer_type == 'character_repetition':
                    word_annotation = f"{word}-{word}-{word}..."  # Show character repetition
                else:
                    word_annotation = f"{word}..."  # Generic stammering
            # Add acoustic mumbling detection
            elif is_unclear and not is_filler:
                word_annotation = f"[{word}?]"  # Unclear/mumbled word
            # Add filler annotation (lower priority)
            elif is_filler:
                word_annotation = f"({word})"  # Regular filler
            
            # Add pitch annotation
            if pitch_level and pitch_level != "NORMAL":
                word_annotation = f"{word_annotation}[{pitch_level}]"
            
            # Add voice quality annotation
            if voice_quality and voice_quality != "normal":
                word_annotation = f"{word_annotation}[{voice_quality}]"
            
            current_line += word_annotation + " "
            last_end_time = end
        
        # Add final line
        if current_line:
            transcript_lines.append(current_line)
        
        return "\n".join(transcript_lines)
    
    def _get_pitch_at_time(self, pitch_data: Dict, time: float) -> Optional[str]:
        """Get pitch level at specific time"""
        variations = pitch_data.get("pitch_variations", [])
        
        # Find closest pitch point
        for var in variations:
            if abs(var["time"] - time) < 0.5:  # Within 0.5 seconds
                return var["level"]
        
        return None
    
    def _get_voice_quality_at_time(self, audio_data, sr: int, start: float, end: float) -> Optional[str]:
        """Get voice quality (energy) at specific time"""
        if audio_data is None:
            return None
        
        try:
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            if start_sample < len(audio_data) and end_sample <= len(audio_data):
                segment = audio_data[start_sample:end_sample]
                energy = np.mean(np.abs(segment))
                
                # Classify energy
                if energy < 0.01:
                    return "low voice"
                elif energy > 0.15:
                    return "loud"
                else:
                    return "normal"
        except:
            pass
        
        return None
    
    def analyze_audio_comprehensive(self, audio_path: Path) -> Dict:
        """
        Comprehensive audio analysis with all features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info(f"Starting comprehensive audio analysis: {audio_path}")
            
            # 1. Transcribe with word timestamps
            transcription_data = self.transcribe_with_word_timestamps(audio_path)
            
            # 2. Detect precise pauses from audio
            pauses = self.detect_precise_pauses(audio_path)
            # Ensure pauses is always a list (never None)
            if pauses is None:
                pauses = []
            logger.info(f"Detected {len(pauses)} pauses from audio analysis")
            
            # FALLBACK 1: If no pauses detected from audio, extract from Whisper segments
            if not pauses or len(pauses) == 0:
                logger.warning("No pauses detected from audio analysis, trying Whisper segment gaps as fallback")
                pauses = self._extract_pauses_from_whisper_segments(transcription_data.get("segments", []))
                logger.info(f"Extracted {len(pauses)} pauses from Whisper segment gaps")
            
            # FALLBACK 2: If still no pauses, extract from word timing gaps (most reliable)
            if not pauses or len(pauses) == 0:
                logger.warning("No pauses from segments either, trying word timing gaps as final fallback")
                pauses = self._extract_pauses_from_word_timing(transcription_data.get("words_with_timing", []))
                logger.info(f"Extracted {len(pauses)} pauses from word timing gaps")
            
            # 3. Track pitch variations
            pitch_data = self.track_pitch_variations(audio_path)
            
            # Load audio once for duration calculations
            # Load audio once for duration calculations - SAFE LOAD
            audio_signal, sr = self._load_audio_safe(audio_path, target_sr=16000)
            total_duration = len(audio_signal) / sr if sr else 0
            
            # 4. Create detailed transcript
            enhanced_transcript = self.create_detailed_transcript(
                transcription_data["words_with_timing"],
                pauses,
                pitch_data,
                audio_path
            )
            
            # 5. Analyze filler words (with acoustic analysis if audio available)
            filler_analysis = self._analyze_fillers(
                transcription_data["transcript"],
                words_with_timing=transcription_data.get("words_with_timing", []),
                audio_path=audio_path,
                total_duration=total_duration
            )
            
            # 6. Calculate speaking metrics
            speaking_metrics = self._calculate_speaking_metrics(
                transcription_data,
                pauses,
                total_duration
            )
            
            # 7. Calculate voice confidence score using improved algorithm
            voice_confidence_data = self._calculate_voice_confidence(
                speaking_metrics,
                filler_analysis,
                pauses,
                pitch_data
            )
            
            # Extract numeric score for backwards compatibility
            voice_confidence_score = voice_confidence_data.get("final_score", 50.0)
            
            # DEBUG: Log voice confidence calculation for verification
            logger.info(f"[VOICE CONFIDENCE CALCULATED] Score: {voice_confidence_score:.1f}/100")
            logger.info(f"  - Base score: {voice_confidence_data.get('base_score', 'N/A')}")
            logger.info(f"  - Category: {voice_confidence_data.get('category', 'N/A')}")
            adjustments = voice_confidence_data.get('adjustments', [])
            for adj in adjustments[:3]:  # Log first 3 adjustments
                logger.info(f"  - {adj.get('factor', 'Unknown')}: {adj.get('points', 0):+.0f} pts ({adj.get('reason', 'N/A')[:50]})")
            
            # Extract filler events from analysis
            text_model_events = filler_analysis.get("text_model_fillers", [])
            acoustic_events = filler_analysis.get("acoustic_events", [])
            # Combine all filler events for trend analysis
            all_filler_events = text_model_events + acoustic_events
            
            # Ensure pauses is a valid list before computing cadence
            valid_pauses = [p for p in pauses if isinstance(p, dict) and isinstance(p.get("duration"), (int, float)) and p.get("duration", 0) > 0]
            logger.info(f"Computing pause cadence with {len(valid_pauses)} valid pauses (from {len(pauses)} total)")
            
            pause_cadence_result = compute_pause_cadence(valid_pauses)
            logger.info(f"Pause cadence computed: {pause_cadence_result}")
            
            advanced_audio_metrics = {
                "filler_trend": compute_filler_trend(all_filler_events, total_duration),
                "pause_cadence": pause_cadence_result,
                "opening_confidence": compute_opening_confidence(
                    all_filler_events,
                    valid_pauses,
                    voice_confidence_score  # Use numeric score here
                )
            }
            
            # Add transcription quality info
            transcription_quality = transcription_data.get("transcription_quality", {})
            
            # 8. Analyze sentence openers for variety and repetition
            sentence_opener_analysis = analyze_sentence_openers(
                transcription_data["transcript"],
                words_with_timing=transcription_data.get("words_with_timing", [])
            )
            logger.info(f"Sentence opener analysis: Status={sentence_opener_analysis.get('status')}, "
                       f"Variety Score={sentence_opener_analysis.get('variety_score')}/100")
            
            logger.info("Comprehensive audio analysis completed")
            
            return {
                "transcript": transcription_data["transcript"],
                "enhanced_transcript": enhanced_transcript,
                "pauses": pauses,
                "pause_summary": {
                    "total_pauses": len(pauses),
                    "long_pauses": len([p for p in pauses if p.get("type") == "LONG_PAUSE"]),
                    "short_pauses": len([p for p in pauses if p.get("type") == "SHORT_PAUSE"]),
                    "medium_pauses": len([p for p in pauses if p.get("duration", 0) >= MEDIUM_PAUSE_THRESHOLD and p.get("duration", 0) < LONG_PAUSE_THRESHOLD]),
                    "total_pause_time": sum(p.get("duration", 0) for p in pauses),
                    "longest_pause": max([p.get("duration", 0) for p in pauses]) if pauses else 0,
                    "avg_pause_duration": round(sum(p.get("duration", 0) for p in pauses) / len(pauses), 2) if pauses else 0
                },
                "pitch_data": pitch_data,
                "filler_analysis": filler_analysis,
                "speaking_metrics": speaking_metrics,
                "voice_confidence_score": voice_confidence_score,
                "voice_confidence_breakdown": voice_confidence_data,  # Full breakdown for transparency
                "words_with_timing": transcription_data["words_with_timing"],
                "advanced_audio_metrics": advanced_audio_metrics,
                "transcription_quality": transcription_quality,
                "sentence_opener_analysis": sentence_opener_analysis,  # NEW: Sentence opener detection
                "analysis_successful": True
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive audio analysis: {e}", exc_info=True)
            # CRITICAL: Even on error, return a meaningful voice_confidence_score
            logger.warning(f"[VOICE CONFIDENCE ERROR] Returning fallback score due to: {str(e)}")
            return {
                "transcript": "",
                "enhanced_transcript": "",
                "pauses": [],
                "voice_confidence_score": 25.0,  # Low score indicates analysis failure, not 0
                "voice_confidence_error": str(e),
                "analysis_successful": False,
                "error": str(e)
            }
    
    def _analyze_fillers(
        self,
        transcript: str,
        words_with_timing: Optional[List[Dict]] = None,
        audio_path: Optional[Path] = None,
        total_duration: Optional[float] = None
    ) -> Dict:
        """
        Analyze filler words using both text and acoustic analysis
        
        Args:
            transcript: Speech transcript
            words_with_timing: Optional word-level timing data for acoustic analysis
            audio_path: Optional audio path for acoustic analysis
        """
        # EXTENSIVE filler word list including ALL mumbling patterns
        filler_words_list = [
            # Basic fillers
            "um", "uh", "ah", "eh", "er", "hmm", "erm", 
            # Extended mumbling patterns (with variations)
            "umm", "uhh", "ahh", "ehh", "err", "uhm", "ahm", "ehm",
            "ummm", "uhhh", "ahhh", "ehhh", "errr",
            # Mumbling with hyphens
            "um-um", "uh-uh", "ah-ah", "er-er", "uh-huh", "mm-hmm", "mhm",
            # Common fillers
            "like", "you know", "well", "actually", "basically", "literally", 
            "kinda", "sorta", "gonna", "wanna", "gotta",
            # Stuttering patterns (repeated syllables)
            "a-a-a", "i-i-i", "e-e-e", "o-o-o", "u-u-u",
            "but-but", "and-and", "the-the", "i-i", "a-a", 
            # Additional mumbling sounds
            "mm", "mmm", "hmm", "hm", "mh", "mhm", "uh-huh", "yeah", "yep", "yup"
        ]
        valid_filler_tokens = {fw.lower().strip('.,!?;:') for fw in filler_words_list}
        
        transcript_text = transcript or ""
        words = transcript_text.lower().split()
        total_words = len(words)
        filler_count = 0
        stutter_count = 0
        filler_positions: List[int] = []
        stutter_positions = []
        filler_breakdown: Counter = Counter()
        acoustic_filler_count = 0
        acoustic_events: List[Dict] = []
        text_model_events: List[Dict] = []
        mumbling_instances: List[Dict] = []
        text_model_error: Optional[str] = None
        
        # Run Hugging Face disfluency detector for advanced text-based filler detection
        text_model_results = {
            "fillers": [],
            "filler_counts": {},
            "filler_ratio": 0.0,
            "alignments": []
        }
        if transcript_text.strip():
            try:
                text_model_results = disfluency_detector.detect(
                    transcript_text,
                    words_with_timing=words_with_timing
                )
            except ImportError as err:
                text_model_error = f"transformers not available: {err}"
                logger.warning(text_model_error)
            except Exception as err:  # pragma: no cover - defensive
                text_model_error = f"Disfluency detector failed: {err}"
                logger.error(text_model_error)
        
        if text_model_error is None:
            text_model_error = text_model_results.get("error")
        
        text_model_ratio = float(text_model_results.get("filler_ratio") or 0.0)
        raw_text_fillers = text_model_results.get("fillers") or []
        text_model_alignments = text_model_results.get("alignments") or []

        filtered_text_model_fillers = []
        for filler_entry in raw_text_fillers:
            token_original = (filler_entry.get("token") or "").strip()
            if not token_original:
                continue
            token_clean = token_original.lower().strip().strip('.,!?;:')
            if not token_clean or token_clean not in valid_filler_tokens:
                continue
            validated = False
            word_index = filler_entry.get("word_index")
            if words_with_timing and isinstance(word_index, int):
                if 0 <= word_index < len(words_with_timing):
                    recognized_word = (words_with_timing[word_index].get("word") or "").lower().strip('.,!?;:')
                    if recognized_word in valid_filler_tokens:
                        validated = True
            if not validated:
                char_start = filler_entry.get("char_start")
                char_end = filler_entry.get("char_end")
                if isinstance(char_start, int):
                    if not isinstance(char_end, int) or char_end <= char_start:
                        candidate = transcript_text[char_start:char_start + len(token_original)]
                    else:
                        candidate = transcript_text[char_start:char_end]
                    candidate_clean = candidate.lower().strip().strip('.,!?;:') if candidate else ""
                    if candidate_clean in valid_filler_tokens:
                        validated = True
            if not validated:
                continue
            entry_copy = dict(filler_entry)
            entry_copy["_clean_token"] = token_clean
            entry_copy["_token_original"] = token_original
            filtered_text_model_fillers.append(entry_copy)

        text_model_fillers_raw = filtered_text_model_fillers
        text_model_counts = Counter(entry["_clean_token"] for entry in text_model_fillers_raw)
        filler_breakdown.update(text_model_counts)
        text_model_counts_original = {token: int(count) for token, count in text_model_counts.items()}
        text_model_filler_count = len(text_model_fillers_raw)
        filler_count = text_model_filler_count
        
        # Map alignments by token/char for quick lookup
        alignment_lookup_char: Dict[Tuple[str, Optional[int]], List[Dict]] = {}
        alignment_lookup_word: Dict[Tuple[str, Optional[int]], List[Dict]] = {}
        for alignment in text_model_alignments:
            key = (str(alignment.get("token", "")).lower(), alignment.get("char_start"))
            alignment_lookup_char.setdefault(key, []).append(alignment)
            if alignment.get("word_index") is not None:
                word_key = (str(alignment.get("token", "")).lower(), alignment.get("word_index"))
                alignment_lookup_word.setdefault(word_key, []).append(alignment)
        
        # Build text-model events with timestamps/positions when possible
        char_word_cache: Dict[int, Optional[int]] = {}
        consumed_tokens = Counter()
        for filler_entry in text_model_fillers_raw:
            token_original = filler_entry.get("_token_original") or filler_entry.get("token", "")
            token_clean = filler_entry.get("_clean_token") or token_original.lower().strip().strip('.,!?;:')
            token = token_clean
            char_start = filler_entry.get("char_start")
            char_end = filler_entry.get("char_end")
            
            alignment = None
            if char_start is not None:
                key = (token, char_start)
                if key in alignment_lookup_char and alignment_lookup_char[key]:
                    alignment = alignment_lookup_char[key].pop(0)
            if alignment is None and filler_entry.get("word_index") is not None:
                key_word = (token, filler_entry.get("word_index"))
                if key_word in alignment_lookup_word and alignment_lookup_word[key_word]:
                    alignment = alignment_lookup_word[key_word].pop(0)
            
            if char_start is not None and char_start not in char_word_cache:
                char_word_cache[char_start] = self._char_index_to_word_position(transcript_text, char_start)
            word_index = char_word_cache.get(char_start) if char_start is not None else None

            if word_index is None and filler_entry.get("word_index") is not None:
                word_index = filler_entry.get("word_index")
            
            start_time = alignment.get("start") if alignment else None
            end_time = alignment.get("end") if alignment else None
            if alignment and alignment.get("word_index") is not None and word_index is None:
                word_index = alignment.get("word_index")
            
            if words_with_timing and word_index is not None and 0 <= word_index < len(words_with_timing):
                word_info = words_with_timing[word_index]
                if start_time is None:
                    start_time = word_info.get("start")
                if end_time is None:
                    end_time = word_info.get("end")
            elif start_time is None and alignment and alignment.get("segment_text"):
                # Keep segment-level timing even without precise word alignment
                start_time = alignment.get("start")
                end_time = alignment.get("end")
            elif start_time is None and total_duration:
                char_position = char_start if isinstance(char_start, (int, float)) else 0
                approx_ratio = max(0.0, min(1.0, char_position / max(1, len(transcript_text))))
                start_time = approx_ratio * float(total_duration)
                end_time = start_time + 0.35
            
            if start_time is None and total_duration:
                char_position = char_start if isinstance(char_start, (int, float)) else 0
                approx_ratio = max(0.0, min(1.0, char_position / max(1, len(transcript_text))))
                start_time = approx_ratio * float(total_duration)
            if start_time is None:
                start_time = 0.0
            if end_time is None:
                end_time = start_time + 0.35
            
            if word_index is not None:
                filler_positions.append(word_index)
            
            event = {
                "label": token,
                "token_original": token_original,
                "char_start": char_start,
                "char_end": char_end,
                "word_index": word_index,
                "start": start_time,
                "end": end_time,
                "method": "text_model",
                "score": float(filler_entry.get("score", 0.0))
            }
            if alignment and alignment.get("segment_text"):
                event["segment_text"] = alignment.get("segment_text")
            text_model_events.append(event)
            consumed_tokens[token] += 1
            
            # Track mumbling events from text model
            if token in valid_filler_tokens or token.replace("-", "") in valid_filler_tokens:
                mumbling_instances.append({
                    "word": token_original or token,
                    "position": word_index,
                    "timestamp": start_time,
                    "type": "mumbling",
                    "detection_method": "text_model",
                    "confidence": event["score"]
                })
        
        if not filler_breakdown and text_model_events:
            for event in text_model_events:
                key = event.get("label")
                if not key:
                    continue
                filler_breakdown[key] += 1
        
        # Build word interval index for quick overlap checks
        word_intervals: List[Tuple[float, float]] = []
        if words_with_timing:
            for word_info in words_with_timing:
                start = word_info.get("start")
                end = word_info.get("end")
                if start is None or end is None:
                    continue
                start = float(start)
                end = float(end)
                if end > start:
                    word_intervals.append((start, end))
        word_intervals.sort(key=lambda iv: iv[0])
        
        # Enhanced stammering/mumbling detection with timestamps
        stammering_instances = []  # Track stammering with timestamps
        # mumbling_instances already initialized earlier with text-model detections
        
        # Track remaining text-model detections so we do not double count
        remaining_text_tokens = Counter(consumed_tokens)
        
        # Text-based filler detection with timestamp tracking (fallback / additional cases)
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?;:')
            word_timestamp = None
            
            if words_with_timing and i < len(words_with_timing):
                word_timestamp = words_with_timing[i].get("start", 0)
            elif words_with_timing:
                for word_info in words_with_timing:
                    if word_info.get("word", "").strip().lower() == clean_word:
                        word_timestamp = word_info.get("start", 0)
                        break
            
            if remaining_text_tokens.get(clean_word, 0) > 0:
                remaining_text_tokens[clean_word] -= 1
                continue
            
            if clean_word in valid_filler_tokens:
                filler_count += 1
                filler_positions.append(i)
                filler_breakdown[clean_word] += 1
                
                mumbling_patterns = [
                    "um", "uh", "ah", "eh", "er", "hmm", "erm", 
                    "umm", "uhh", "ahh", "ehh", "err", "uhm", "ahm", "ehm",
                    "ummm", "uhhh", "ahhh", "ehhh", "errr",
                    "um-um", "uh-uh", "ah-ah", "er-er", 
                    "uh-huh", "mm-hmm", "mhm", "mm", "mmm", "hm", "mh",
                    "yeah", "yep", "yup"
                ]
                if clean_word in mumbling_patterns:
                    mumbling_instances.append({
                        'word': clean_word,
                        'position': i,
                        'timestamp': word_timestamp,
                        'type': 'mumbling',
                        'detection_method': 'text_fallback'
                    })
                
                for pattern in mumbling_patterns:
                    if pattern in clean_word and clean_word not in mumbling_patterns:
                        mumbling_instances.append({
                            'word': clean_word,
                            'position': i,
                            'timestamp': word_timestamp,
                            'type': 'mumbling',
                            'matched_pattern': pattern,
                            'detection_method': 'text_fallback_partial'
                        })
                        break
            
            # Enhanced stuttering detection (repeated characters, words, syllables)
            if i > 0:
                prev_word = words[i-1].strip('.,!?;:')
                is_stutter = False
                stutter_type = None
                
                # Single character repetition (a-a-a, i-i-i)
                if len(clean_word) == 1 and len(prev_word) == 1 and clean_word == prev_word:
                    is_stutter = True
                    stutter_type = 'character_repetition'
                # Word repetition (the-the, and-and)
                elif clean_word == prev_word and len(clean_word) > 1:
                    is_stutter = True
                    stutter_type = 'word_repetition'
                # Syllable repetition pattern (detect common patterns)
                elif len(clean_word) >= 2 and len(prev_word) >= 2:
                    # Check for repeated syllables (e.g., "but-but", "um-um")
                    if clean_word == prev_word and '-' in clean_word:
                        is_stutter = True
                        stutter_type = 'syllable_repetition'
                
                if is_stutter:
                    stutter_count += 1
                    stutter_positions.append(i)
                    stammering_instances.append({
                        'word': clean_word,
                        'position': i,
                        'timestamp': word_timestamp,
                        'type': stutter_type,
                        'previous_word': prev_word
                    })
        
        # Acoustic-based filler/mumbling detection (if audio available)
        # DISABLED BY DEFAULT - generates too many false positives
        if not DISABLE_ACOUSTIC_FILLER_DETECTION and audio_path and words_with_timing:
            try:
                y, sr = librosa.load(str(audio_path), sr=16000)
                
                for word_info in words_with_timing:
                    word = word_info.get("word", "").strip().lower()
                    start = word_info.get("start", 0)
                    end = word_info.get("end", 0)
                    
                    # Extract audio segment for this word
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    
                    if start_sample < len(y) and end_sample <= len(y) and end_sample > start_sample:
                        segment = y[start_sample:end_sample]
                        
                        metrics = self._detect_filler_acoustic(segment, sr)
                        if metrics:
                            # STRICT THRESHOLD: Use configurable confidence minimum
                            confidence = float(metrics.get("confidence", 0.0))
                            rms = float(metrics.get("avg_rms", 0.0))
                            
                            # Must have HIGH confidence AND actual audio energy (not silence)
                            if confidence < ACOUSTIC_FILLER_MIN_CONFIDENCE:
                                continue
                            if rms < ACOUSTIC_FILLER_MIN_RMS:
                                continue  # Skip if audio is too quiet (likely silence, not real filler)
                                
                            clean_word = word.strip('.,!?;:')
                            if clean_word and clean_word not in valid_filler_tokens:
                                continue

                            classification = None
                            event_label = clean_word or None
                            if not event_label:
                                classification = self._classify_acoustic_filler(metrics, {
                                    "source": "word_segment",
                                    "previous_word": word_info.get("word", "")
                                })
                                event_label = classification.get("label", "murmur")
                            raw_label = clean_word or 'murmur'

                            event = {
                                'label': event_label,
                                'raw_label': raw_label,
                                'start': float(start),
                                'end': float(end),
                                'duration': float(max(0.0, end - start)),
                                'method': 'word_segment',
                                'confidence': float(metrics.get('confidence', 0.6)),
                                'features': {
                                    'avg_zcr': float(metrics.get('avg_zcr', 0.0)),
                                    'avg_centroid': float(metrics.get('avg_centroid', 0.0)),
                                    'avg_rms': float(metrics.get('avg_rms', 0.0)),
                                    'avg_rolloff': float(metrics.get('avg_rolloff', 0.0))
                                }
                            }
                            if classification:
                                event['classification'] = classification
                            acoustic_events.append(event)
                            
                            # Only treat as filler if we have a known filler token
                            if event_label in valid_filler_tokens:
                                acoustic_filler_count += 1
                                filler_breakdown[f"[acoustic] {event_label}"] += 1
                                filler_count += 1

                            mumbling_instances.append({
                                'word': event_label,
                                'position': len(words),
                                'timestamp': start,
                                'type': 'mumbling',
                                'detection_method': 'acoustic_segment',
                                'confidence': event['confidence'],
                                'classifier_confidence': (classification or {}).get('confidence')
                            })
                            logger.debug(f"Acoustically detected filler/mumbling: {event_label} at {start:.2f}s")
                
                if acoustic_filler_count:
                    logger.info(f"Acoustic filler detection: {acoustic_filler_count} additional fillers/mumbling detected")
                
                # Detect short murmurs between consecutive words (e.g., "uh" in small gaps)
                for idx in range(len(words_with_timing) - 1):
                    current_word = words_with_timing[idx]
                    next_word = words_with_timing[idx + 1]
                    current_end = current_word.get("end", 0.0)
                    next_start = next_word.get("start", 0.0)
                    gap = max(0.0, float(next_start) - float(current_end))
                    
                    if 0.08 <= gap <= 0.6:
                        gap_start_sample = int(current_end * sr)
                        gap_end_sample = int(next_start * sr)
                        if gap_end_sample > gap_start_sample and gap_end_sample <= len(y):
                            gap_segment = y[gap_start_sample:gap_end_sample]
                            metrics = self._detect_filler_acoustic(gap_segment, sr)
                            if metrics:
                                # STRICT THRESHOLD: Use configurable confidence + RMS check
                                confidence = float(metrics.get("confidence", 0.0))
                                rms = float(metrics.get("avg_rms", 0.0))
                                
                                # Must have HIGH confidence AND actual audio energy (not silence)
                                if confidence < ACOUSTIC_FILLER_MIN_CONFIDENCE:
                                    continue
                                if rms < ACOUSTIC_FILLER_MIN_RMS:
                                    continue  # Silence between words, not a filler
                                    
                                classification = self._classify_acoustic_filler(metrics, {
                                    "source": "between_words",
                                    "previous_word": current_word.get("word", ""),
                                    "next_word": next_word.get("word", "")
                                })
                                classified_label = classification.get("label", "murmur")
                                event = {
                                    "label": classified_label,
                                    "raw_label": "murmur",
                                    "start": float(current_end),
                                    "end": float(next_start),
                                    "duration": float(gap),
                                    "method": "between_words",
                                    "confidence": float(metrics.get("confidence", 0.6)),
                                    "features": {
                                        'avg_zcr': float(metrics.get('avg_zcr', 0.0)),
                                        'avg_centroid': float(metrics.get('avg_centroid', 0.0)),
                                        'avg_rms': float(metrics.get('avg_rms', 0.0)),
                                        'avg_rolloff': float(metrics.get('avg_rolloff', 0.0))
                                    },
                                    "context": {
                                        "previous_word": current_word.get("word", ""),
                                        "next_word": next_word.get("word", "")
                                    }
                                }
                                event["classification"] = classification
                                acoustic_events.append(event)

                                if classified_label in valid_filler_tokens:
                                    acoustic_filler_count += 1
                                    filler_breakdown[f"[acoustic] {classified_label}"] += 1
                                    filler_count += 1
                                mumbling_instances.append({
                                    "word": classified_label,
                                    "position": idx,
                                    "timestamp": current_end,
                                    "type": "mumbling",
                                    "detection_method": "between_words",
                                    "confidence": event["confidence"],
                                    "classifier_confidence": classification.get("confidence")
                                })
                                logger.debug(f"Gap murmur detected between '{current_word.get('word', '')}' and '{next_word.get('word', '')}' at {current_end:.2f}s")
            except Exception as e:
                logger.warning(f"Acoustic filler detection failed: {e}")
        
        # Window-based acoustic scan to catch murmurs missed by transcript
        # DISABLED BY DEFAULT - generates too many false positives
        if not DISABLE_ACOUSTIC_FILLER_DETECTION and audio_path:
            try:
                window_events = self._detect_unaligned_acoustic_fillers(audio_path, word_intervals)
                if window_events:
                    for event in window_events:
                        classification_metrics = dict(event.get("features", {}))
                        classification_metrics["duration"] = event.get("duration", classification_metrics.get("duration", 0.0))
                        classification = self._classify_acoustic_filler(classification_metrics, {
                            "source": event.get("method", "window_scan")
                        })
                        classified_label = classification.get("label", event.get("label", "murmur"))
                        event["raw_label"] = event.get("label", "murmur")
                        event["label"] = classified_label
                        event["classification"] = classification
                        acoustic_events.append(event)
                        acoustic_filler_count += 1
                        filler_count += 1
                        breakdown_key = f"[acoustic] {classified_label}"
                        filler_breakdown[breakdown_key] += 1
                        mumbling_instances.append({
                            'word': classified_label,
                            'position': len(words),
                            'timestamp': event.get('start', 0.0),
                            'type': 'mumbling',
                            'detection_method': event.get('method', 'window_scan'),
                            'confidence': event.get('confidence', 0.6),
                            'classifier_confidence': classification.get("confidence"),
                            'duration': event.get('duration', 0.0)
                        })
                    logger.info("Windowed acoustic filler detection: %d segments", len(window_events))
            except Exception as e:
                logger.warning(f"Windowed acoustic filler detection failed: {e}")
        
        # Also detect mumbling from pauses (long pauses might indicate hesitation/mumbling)
        if words_with_timing and len(words_with_timing) > 1:
            for i in range(len(words_with_timing) - 1):
                current_word = words_with_timing[i]
                next_word = words_with_timing[i + 1]
                gap = next_word.get("start", 0) - current_word.get("end", 0)
                
                # If there's a significant gap (>0.5s) between words, might indicate mumbling/hesitation
                if gap > 0.5 and gap < 2.0:  # Short to medium pause
                    # Check if the word before the pause is unclear or short
                    word_text = current_word.get("word", "").strip().lower()
                    if len(word_text) <= 2 or word_text in ["a", "i", "uh", "um"]:
                        mumbling_instances.append({
                            'word': word_text or '[pause]',
                            'position': i,
                            'timestamp': current_word.get("end", 0),
                            'type': 'mumbling',
                            'detection_method': 'pause_based',
                            'pause_duration': gap
                        })
        
        stutter_ratio = stutter_count / total_words if total_words > 0 else 0
        
        # Group mumbling/stammering by time windows for better visualization
        mumbling_clusters = self._cluster_by_time(mumbling_instances, window_seconds=2.0)
        stammering_clusters = self._cluster_by_time(stammering_instances, window_seconds=2.0)
        
        return {
            "total_fillers": filler_count,
            "filler_ratio": filler_count / total_words if total_words > 0 else 0,
            "filler_breakdown": dict(filler_breakdown),
            "filler_positions": filler_positions,
            "acoustic_fillers": acoustic_filler_count,
            "acoustic_filler_count": acoustic_filler_count,
            "acoustic_events": acoustic_events,
            "text_model_fillers": text_model_events,
            "text_model_filler_count": text_model_filler_count,
            "text_model_ratio": text_model_ratio,
            "text_model_error": text_model_error,
            "text_model_counts": text_model_counts_original,
            "stutter_count": stutter_count,
            "stutter_ratio": stutter_ratio,
            "stutter_positions": stutter_positions,
            "stammering_detected": stutter_count > 3 or stutter_ratio > 0.05,
            
            # NEW: Precise stammering/mumbling tracking with timestamps
            "mumbling_instances": mumbling_instances,  # All mumbling with timestamps
            "stammering_instances": stammering_instances,  # All stammering with timestamps
            "mumbling_clusters": mumbling_clusters,  # Grouped mumbling moments
            "stammering_clusters": stammering_clusters,  # Grouped stammering moments
            "mumbling_count": len(mumbling_instances),
            "stammering_count": len(stammering_instances)
        }
    
    def _cluster_by_time(self, instances: List[Dict], window_seconds: float = 2.0) -> List[Dict]:
        """
        Cluster instances by time windows for better visualization
        
        Args:
            instances: List of instances with timestamps
            window_seconds: Time window for clustering
            
        Returns:
            List of clusters with start/end times and counts
        """
        if not instances:
            return []
        
        # Sort by timestamp
        sorted_instances = sorted([i for i in instances if i.get('timestamp') is not None], 
                                 key=lambda x: x.get('timestamp', 0))
        
        if not sorted_instances:
            return []
        
        clusters = []
        current_cluster = {
            'start_time': sorted_instances[0].get('timestamp', 0),
            'end_time': sorted_instances[0].get('timestamp', 0),
            'instances': [sorted_instances[0]],
            'count': 1
        }
        
        for instance in sorted_instances[1:]:
            timestamp = instance.get('timestamp', 0)
            # If within window, add to current cluster
            if timestamp - current_cluster['end_time'] <= window_seconds:
                current_cluster['end_time'] = timestamp
                current_cluster['instances'].append(instance)
                current_cluster['count'] += 1
            else:
                # Start new cluster
                clusters.append(current_cluster)
                current_cluster = {
                    'start_time': timestamp,
                    'end_time': timestamp,
                    'instances': [instance],
                    'count': 1
                }
        
        # Add last cluster
        if current_cluster['instances']:
            clusters.append(current_cluster)
        
        return clusters
    
    def _char_index_to_word_position(self, transcript: str, char_index: Optional[int]) -> Optional[int]:
        """
        Estimate the word position within a transcript for a given character index.
        """
        if transcript is None or char_index is None or char_index < 0:
            return None
        prefix = transcript[:char_index]
        if not prefix:
            return 0
        # Split using default whitespace; aligns with earlier lower-casing split
        return len(prefix.split())
    
    def _detect_filler_acoustic(self, audio_segment: np.ndarray, sr: int) -> Optional[Dict]:
        """
        Detect filler words/mumbling from acoustic characteristics
        
        Args:
            audio_segment: Audio segment to analyze
            sr: Sample rate
            
        Returns:
            Dict of metrics if the segment has filler-like characteristics, otherwise None
        """
        try:
            if len(audio_segment) == 0:
                return None
            
            # Calculate acoustic features
            # 1. Zero-crossing rate (fillers often have high ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio_segment)[0]
            avg_zcr = np.mean(zcr)
            
            # 2. Spectral centroid (fillers are often lower frequency)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroid)
            
            # 3. RMS energy (fillers are often quieter)
            rms = librosa.feature.rms(y=audio_segment)[0]
            avg_rms = np.mean(rms)
            # Skip pure silence, but allow quiet filler sounds (uh, ah can be soft)
            # Lowered from 0.012 to 0.003 to catch quiet fillers
            if avg_rms < 0.003:
                return None
            
            # 4. Duration (fillers are often short)
            duration = len(audio_segment) / sr
            
            # 5. Spectral rolloff (fillers have lower rolloff)
            rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)[0]
            avg_rolloff = np.mean(rolloff)
            
            # Heuristic: Filler words typically have:
            # - High ZCR (>0.1)
            # - Low spectral centroid (<2000 Hz)
            # - Low RMS energy
            # - Short duration (<0.5s)
            # - Low spectral rolloff
            
            is_filler = False
            confidence = 0.0
            
            # Check multiple conditions
            if duration < 0.6:  # Short duration
                # Relaxed thresholds to catch real fillers like "uh", "ah"
                high_zcr = avg_zcr > 0.08  # Lowered from 0.12
                low_centroid = avg_centroid < 2800  # Raised from 2200 (fillers can have higher centroid)
                quiet = avg_rms < 0.08  # Raised from 0.06
                very_quiet = avg_rms < 0.04  # Raised from 0.03
                low_rolloff = avg_rolloff < 3500  # Raised from 2800
                has_some_voice = avg_rms > 0.005  # Must have some voice energy
                
                # Require at least two corroborating signals to reduce false positives
                if high_zcr and quiet and has_some_voice:
                    is_filler = True
                elif low_centroid and very_quiet and low_rolloff and has_some_voice:
                    is_filler = True
                elif low_centroid and quiet and has_some_voice:
                    # Also catch voiced sounds with low centroid (typical of "uh", "ah")
                    is_filler = True
            
            if is_filler:
                confidence_components = [
                    min(1.0, avg_zcr / 0.2),
                    max(0.0, 1.0 - min(avg_centroid / 4000, 1.0)),
                    min(1.0, (avg_rms / 0.05) if avg_rms > 0 else 0.0)
                ]
                confidence = float(np.clip(np.mean(confidence_components), 0.2, 0.99))
                return {
                    "avg_zcr": float(avg_zcr),
                    "avg_centroid": float(avg_centroid),
                    "avg_rms": float(avg_rms),
                    "avg_rolloff": float(avg_rolloff),
                    "duration": float(duration),
                    "confidence": confidence
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error in acoustic filler detection: {e}")
            return None

    def _classify_acoustic_filler(self, metrics: Dict, context: Optional[Dict] = None) -> Dict:
        """
        Infer a human-readable filler token (uh, umm, ahh, etc.) from acoustic metrics.
        This provides a better UX than the generic "murmur" label while remaining lightweight.
        """
        duration = float(metrics.get("duration") or metrics.get("segment_duration") or 0.0)
        centroid = float(metrics.get("avg_centroid") or 0.0)
        rolloff = float(metrics.get("avg_rolloff") or 0.0)
        zcr = float(metrics.get("avg_zcr") or 0.0)
        rms = float(metrics.get("avg_rms") or 0.0)

        # If confidence signals are weak, stick with neutral "murmur"
        if zcr < 0.16 or rms < 0.015:
            return {
                "label": "murmur",
                "confidence": metrics.get("confidence", 0.3),
                "details": metrics,
                "reason": "insufficient acoustic evidence for explicit filler syllable"
            }

        # Duration-driven base syllable length (only when confident)
        if duration <= 0.18:
            length_variant = "uh"
        elif duration <= 0.32:
            length_variant = "uhh"
        elif duration <= 0.48:
            length_variant = "umm"
        elif duration <= 0.7:
            length_variant = "ummm"
        else:
            length_variant = "aaaa"

        # Spectral centroid nudges vowel quality
        if centroid < 950:
            base = "mm"
        elif centroid < 1500:
            base = "umm"
        elif centroid < 2100:
            base = "uhh"
        elif centroid < 2700:
            base = "ahh"
        else:
            base = "uh"

        # Combine heuristics: prefer the more "closed" sound for soft/low rolloff murmurs
        if rolloff < 2200 or rms < 0.02:
            candidate = base if base.startswith("mm") or base.startswith("um") else length_variant
        else:
            candidate = length_variant if length_variant.startswith("uh") else base

        # Mild smoothing: ensure consistent casing and remove double vowels
        candidate = candidate.lower().replace("aaaa", "ahhh")
        if candidate in {"mm", "mmmm"}:
            candidate = "mm"
        if candidate.endswith("mmmm"):
            candidate = candidate[:-1]

        # Confidence estimate based on how closely the metrics match filler envelope
        duration_score = max(0.0, 1.0 - min(abs(duration - 0.28) / 0.4, 1.0))
        centroid_score = max(0.0, 1.0 - min((centroid - 1800) ** 2 / (1800 ** 2), 1.0))
        zcr_score = min(zcr / 0.25, 1.0)
        rolloff_score = max(0.0, 1.0 - min(rolloff / 7000, 1.0))
        confidence = float(np.clip(np.mean([duration_score, centroid_score, zcr_score, rolloff_score]), 0.25, 0.98))

        classifier_payload = {
            "label": candidate,
            "confidence": confidence,
            "duration": duration,
            "avg_centroid": centroid,
            "avg_rolloff": rolloff,
            "avg_zcr": zcr,
            "avg_rms": rms,
        }

        if context:
            classifier_payload["context"] = context

        return classifier_payload
    
    def _detect_unaligned_acoustic_fillers(self, audio_path: Path, word_intervals: List[Tuple[float, float]], min_gap_overlap: float = 0.1) -> List[Dict]:
        """
        Scan the entire waveform to detect short murmurs that were not aligned to transcript words.
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            if y.size == 0:
                return []
            
            non_silent_intervals = librosa.effects.split(y, top_db=26, frame_length=1024, hop_length=256)
            events: List[Dict] = []
            
            for start_sample, end_sample in non_silent_intervals:
                start_time = start_sample / sr
                end_time = end_sample / sr
                duration = end_time - start_time
                
                # Focus on short bursts that resemble murmurs
                if duration < 0.04 or duration > 1.0:
                    continue
                if self._interval_overlaps_words(start_time, end_time, word_intervals, min_gap_overlap):
                    continue
                
                segment = y[start_sample:end_sample]
                metrics = self._detect_filler_acoustic(segment, sr)
                if not metrics:
                    continue
                
                # STRICT THRESHOLD: Use configurable confidence + RMS check
                confidence = float(metrics.get("confidence", 0.0))
                rms = float(metrics.get("avg_rms", 0.0))
                
                # Must have HIGH confidence AND actual audio energy
                if confidence < ACOUSTIC_FILLER_MIN_CONFIDENCE:
                    continue
                if rms < ACOUSTIC_FILLER_MIN_RMS:
                    continue  # Silence, not a real filler
                
                events.append({
                    "label": "murmur",
                    "start": float(start_time),
                    "end": float(end_time),
                    "duration": float(duration),
                    "method": "window_scan",
                    "confidence": confidence,
                    "features": {
                        "avg_zcr": float(metrics.get("avg_zcr", 0.0)),
                        "avg_centroid": float(metrics.get("avg_centroid", 0.0)),
                        "avg_rms": float(metrics.get("avg_rms", 0.0)),
                        "avg_rolloff": float(metrics.get("avg_rolloff", 0.0))
                    }
                })
            
            return events
        except Exception as e:
            logger.debug(f"Unaligned acoustic filler detection failed: {e}")
            return []
    
    def _interval_overlaps_words(self, start: float, end: float, intervals: List[Tuple[float, float]], max_overlap: float = 0.1) -> bool:
        """Return True if the [start, end] interval overlaps any word interval."""
        for word_start, word_end in intervals:
            if word_end < start:
                continue
            if word_start > end:
                break
            if word_start < end and word_end > start:
                overlap = min(end, word_end) - max(start, word_start)
                if overlap > max_overlap:
                    return True
        return False
    
    def _score_conciseness_from_rate(self, speaking_rate: float) -> float:
        """Map speaking rate (words per minute) to a concise delivery score."""
        if speaking_rate <= 0:
            return 45.0
        if 130 <= speaking_rate <= 160:
            return 92.0
        if 120 <= speaking_rate < 130 or 160 < speaking_rate <= 170:
            return 86.0
        if 105 <= speaking_rate < 120 or 170 < speaking_rate <= 185:
            return 74.0
        if 95 <= speaking_rate < 105 or 185 < speaking_rate <= 200:
            return 62.0
        if 85 <= speaking_rate < 95 or 200 < speaking_rate <= 215:
            return 52.0
        return 45.0
    
    def _calculate_speaking_metrics(self, transcription_data: Dict, pauses: List[Dict], total_duration: float) -> Dict:
        """
        Calculate speaking metrics with MATHEMATICAL VALIDATION
        
        All calculations are verified for accuracy and consistency.
        """
        transcript = transcription_data.get("transcript", "")
        words = transcript.split()
        total_words = len(words)
        
        # Calculate actual speaking time (excluding pauses)
        pause_time = sum(p.get("duration", 0) for p in pauses if p.get("duration"))
        speaking_time = max(0, total_duration - pause_time)
        
        # Calculate speaking rate with validation
        if speaking_time > 0:
            speaking_rate = (total_words / speaking_time) * 60
        else:
            speaking_rate = 0
        
        # VALIDATION: Check for physically plausible WPM
        if VALIDATION_ENABLED:
            if speaking_rate > WPM_MAX_PLAUSIBLE:
                logger.warning(f"WPM {speaking_rate:.1f} exceeds plausible maximum ({WPM_MAX_PLAUSIBLE})")
                speaking_rate = WPM_MAX_PLAUSIBLE  # Cap at maximum
            elif speaking_rate < WPM_MIN_PLAUSIBLE and total_words > 5:
                logger.warning(f"WPM {speaking_rate:.1f} below plausible minimum ({WPM_MIN_PLAUSIBLE})")
        
        # Round to 1 decimal place for consistency
        speaking_rate = round(speaking_rate, 1)
        
        # Categorize speaking rate using configured thresholds
        if speaking_rate < WPM_SLOW_THRESHOLD:
            rate_category = "SLOW"
        elif speaking_rate > WPM_FAST_THRESHOLD:
            rate_category = "FAST"
        else:
            rate_category = "CONVERSATIONAL"
        
        # Calculate pause percentage with validation
        if total_duration > 0:
            pause_percentage = (pause_time / total_duration) * 100
        else:
            pause_percentage = 0
        
        # VALIDATION: Pause percentage cannot exceed 100%
        pause_percentage = min(100.0, max(0.0, pause_percentage))
        pause_percentage = round(pause_percentage, 2)
        
        # Calculate conciseness score
        conciseness_score = self._score_conciseness_from_rate(speaking_rate)
        
        # Categorize pauses
        short_pauses = len([p for p in pauses if p.get("type") == "SHORT_PAUSE"])
        medium_pauses = len([p for p in pauses if p.get("duration", 0) >= MEDIUM_PAUSE_THRESHOLD and p.get("duration", 0) < LONG_PAUSE_THRESHOLD])
        long_pauses = len([p for p in pauses if p.get("type") == "LONG_PAUSE"])
        
        # Calculate average pause duration
        if pauses:
            avg_pause_duration = pause_time / len(pauses)
        else:
            avg_pause_duration = 0
        
        return {
            "total_words": total_words,
            "total_duration": round(total_duration, 2),
            "speaking_time": round(speaking_time, 2),
            "pause_time": round(pause_time, 2),
            "speaking_rate_wpm": speaking_rate,
            "rate_category": rate_category,
            "pause_percentage": pause_percentage,
            "conciseness_score": conciseness_score,
            "total_pauses": len(pauses),
            "short_pauses": short_pauses,
            "medium_pauses": medium_pauses,
            "long_pauses": long_pauses,
            "avg_pause_duration": round(avg_pause_duration, 2),
            # Validation info
            "_validated": True,
            "_wpm_calculation": f"{total_words} words / ({speaking_time:.2f}s / 60) = {speaking_rate} WPM"
        }
    
    def _calculate_voice_confidence(self, speaking_metrics: Dict, filler_analysis: Dict, pauses: List[Dict], pitch_data: Dict) -> Dict:
        """
        Calculate voice confidence score - EVIDENCE-BASED ALGORITHM
        
        Returns a dictionary with score AND breakdown for transparency.
        All adjustments are documented and defensible.
        
        Score ranges:
        - 0-30:  Poor (severe issues)
        - 31-50: Below Average (noticeable issues)
        - 51-70: Average (typical presentation)
        - 71-85: Good (above average presentation)
        - 86-95: Excellent (professional quality)
        - 96-100: Exceptional (rarely achieved)
        """
        try:
            # Initialize score breakdown for transparency
            breakdown = {
                "base_score": 50.0,
                "adjustments": [],
                "final_score": 50.0,
                "category": "Average"
            }
            
            # Start with moderate base score (50 = average)
            score = 50.0
            
            # === SPEAKING RATE (max ±25 points) ===
            speaking_rate = speaking_metrics.get("speaking_rate_wpm", 0)
            rate_adjustment = 0
            rate_reason = ""
            
            if speaking_rate <= 0:
                # Don't penalize too harshly if rate is 0 - might be calculation issue
                # Only penalize if we're sure there's no speech (check total_words)
                total_words = speaking_metrics.get("total_words", 0)
                if total_words == 0:
                    rate_adjustment = -20
                    rate_reason = "No speech detected (0 words)"
                else:
                    # If we have words but rate is 0, it's likely a calculation error
                    # Use a moderate penalty and estimate rate from words/duration
                    total_duration = speaking_metrics.get("total_duration", 0)
                    if total_duration > 0:
                        estimated_rate = (total_words / total_duration) * 60
                        speaking_rate = estimated_rate  # Use estimated rate for scoring
                        if 120 <= estimated_rate <= 180:
                            rate_adjustment = 15
                            rate_reason = f"Estimated rate ({estimated_rate:.0f} WPM from {total_words} words)"
                        else:
                            rate_adjustment = 5
                            rate_reason = f"Estimated rate ({estimated_rate:.0f} WPM) - using fallback calculation"
                    else:
                        rate_adjustment = -10  # Less harsh penalty if we have words
                        rate_reason = "Unable to calculate speaking rate but speech detected"
            else:  # speaking_rate > 0
                # IMPROVED: Smooth linear adjustment instead of discrete steps
                # Formula: speaking_rate_adj = clamp(((WPM - 90) / 70) * 25, -12, +25)
                rate_adjustment = ((speaking_rate - 90) / 70) * 25
                rate_adjustment = max(-12, min(25, rate_adjustment))
                
                # Generate appropriate reason
                if speaking_rate >= 135 and speaking_rate <= 160:
                    rate_reason = f"Excellent speaking pace ({speaking_rate:.0f} WPM, ideal range 135-160)"
                elif speaking_rate >= 110 and speaking_rate < 135:
                    rate_reason = f"Good speaking pace ({speaking_rate:.0f} WPM)"
                elif speaking_rate >= 90 and speaking_rate < 110:
                    rate_reason = f"Slightly slow pace ({speaking_rate:.0f} WPM)"
                elif speaking_rate < 90:
                    rate_reason = f"Slow pace ({speaking_rate:.0f} WPM, below 90) - mild penalty"
                elif speaking_rate > 160 and speaking_rate <= 180:
                    rate_reason = f"Slightly fast pace ({speaking_rate:.0f} WPM)"
                else:  # > 180
                    rate_reason = f"Fast pace ({speaking_rate:.0f} WPM, above 180)"
            
            score += rate_adjustment
            breakdown["adjustments"].append({
                "factor": "Speaking Rate",
                "points": rate_adjustment,
                "reason": rate_reason,
                "evidence": f"{speaking_rate:.1f} WPM"
            })
            
            # === FILLER WORD USAGE (max ±20 points) ===
            filler_ratio = filler_analysis.get("filler_ratio", 0)
            total_fillers = filler_analysis.get("total_fillers", 0)
            filler_adjustment = 0
            filler_reason = ""
            
            # IMPROVED: Continuous scale with cap to avoid over-penalization
            # Formula: filler_voice_adj = clamp(-40 * (filler_ratio - 0.02), -20, +20)
            raw_filler_adj = -40 * (filler_ratio - 0.02)
            filler_adjustment = max(-20, min(20, raw_filler_adj))
            
            if filler_ratio < 0.02:
                filler_reason = f"Excellent filler control ({total_fillers} fillers, {filler_ratio*100:.1f}%)"
            elif filler_ratio < 0.06:
                filler_reason = f"Good filler control ({total_fillers} fillers, {filler_ratio*100:.1f}%)"
            elif filler_ratio < 0.10:
                filler_reason = f"Noticeable fillers ({total_fillers} fillers, {filler_ratio*100:.1f}%)"
            else:
                filler_reason = f"Excessive filler usage ({total_fillers} fillers, {filler_ratio*100:.1f}%) - capped penalty"
            
            score += filler_adjustment
            breakdown["adjustments"].append({
                "factor": "Filler Words",
                "points": filler_adjustment,
                "reason": filler_reason,
                "evidence": f"{filler_ratio*100:.2f}% ({total_fillers} fillers)"
            })
            
            # === PAUSE ANALYSIS (max ±20 points) ===
            long_pause_count = len([p for p in pauses if p.get("type") == "LONG_PAUSE"])
            total_pause_count = len(pauses)
            pause_percentage = speaking_metrics.get("pause_percentage", 0)
            
            # Long pause scoring
            pause_adjustment = 0
            pause_reason = ""
            
            if long_pause_count == 0:
                pause_adjustment += 8
                pause_reason = "No long pauses"
            elif long_pause_count <= 1:
                pause_adjustment += 5
                pause_reason = f"Minimal long pauses ({long_pause_count})"
            elif long_pause_count <= 3:
                pause_adjustment -= 5  # IMPROVED: Softer penalty (-5 instead of old -10)
                pause_reason = f"Few long pauses ({long_pause_count})"
            elif long_pause_count <= 5:
                pause_adjustment -= 5
                pause_reason = f"Several long pauses ({long_pause_count})"
            else:
                pause_adjustment -= 15
                pause_reason = f"Too many long pauses ({long_pause_count})"
            
            # Natural rhythm scoring
            rhythm_adjustment = 0
            rhythm_reason = ""
            
            if 5 <= total_pause_count <= 15:
                rhythm_adjustment = 8
                rhythm_reason = f"Natural speech rhythm ({total_pause_count} pauses)"
            elif 3 <= total_pause_count < 5 or 15 < total_pause_count <= 20:
                rhythm_adjustment = 4
                rhythm_reason = f"Reasonable rhythm ({total_pause_count} pauses)"
            elif total_pause_count < 3 and speaking_metrics.get("total_words", 0) > 50:
                rhythm_adjustment = -5
                rhythm_reason = f"Rushed delivery (only {total_pause_count} pauses)"
            elif total_pause_count > 20:
                rhythm_adjustment = -10
                rhythm_reason = f"Choppy delivery ({total_pause_count} pauses)"
            
            # Silence percentage penalty
            silence_penalty = 0
            if pause_percentage > 30:
                silence_penalty = -10
            elif pause_percentage > 25:
                silence_penalty = -5
            
            total_pause_adjustment = pause_adjustment + rhythm_adjustment + silence_penalty
            score += total_pause_adjustment
            breakdown["adjustments"].append({
                "factor": "Pause Pattern",
                "points": total_pause_adjustment,
                "reason": f"{pause_reason}; {rhythm_reason}",
                "evidence": f"{total_pause_count} pauses, {pause_percentage:.1f}% silence"
            })
            
            # === PITCH STABILITY (max ±15 points) ===
            pitch_std = pitch_data.get("pitch_std", 0)
            avg_pitch = pitch_data.get("avg_pitch", 0)
            
            # Calculate pitch coefficient of variation for normalized comparison
            if avg_pitch > 0:
                pitch_cv = pitch_std / avg_pitch
            else:
                pitch_cv = 0
            
            pitch_adjustment = 0
            pitch_reason = ""
            
            if pitch_cv < 0.10:
                pitch_adjustment = 15
                pitch_reason = "Very stable pitch"
            elif pitch_cv < 0.15:
                pitch_adjustment = 12
                pitch_reason = "Stable pitch"
            elif pitch_cv < 0.20:
                pitch_adjustment = 8
                pitch_reason = "Reasonably stable pitch"
            elif pitch_cv < 0.30:
                pitch_adjustment = 3
                pitch_reason = "Somewhat variable pitch"
            elif pitch_cv < 0.40:
                pitch_adjustment = -5
                pitch_reason = "Unstable pitch variation"
            else:
                pitch_adjustment = -15
                pitch_reason = "Very unstable pitch"
            
            score += pitch_adjustment
            breakdown["adjustments"].append({
                "factor": "Pitch Stability",
                "points": pitch_adjustment,
                "reason": pitch_reason,
                "evidence": f"CV={pitch_cv:.3f}, std={pitch_std:.1f}Hz"
            })
            
            # === STAMMERING/STUTTERING (penalty only) ===
            if filler_analysis.get("stammering_detected", False):
                stutter_count = filler_analysis.get("stutter_count", 0)
                stutter_adjustment = 0
                
                if stutter_count > 10:
                    stutter_adjustment = -25
                elif stutter_count > 5:
                    stutter_adjustment = -15
                elif stutter_count > 2:
                    stutter_adjustment = -8
                
                if stutter_adjustment != 0:
                    score += stutter_adjustment
                    breakdown["adjustments"].append({
                        "factor": "Stammering",
                        "points": stutter_adjustment,
                        "reason": f"Stammering detected ({stutter_count} instances)",
                        "evidence": f"{stutter_count} stammers"
                    })
            
            # === DURATION ADJUSTMENT (short presentations) ===
            total_duration = speaking_metrics.get("total_duration", 0)
            duration_adjustment = 0
            
            if total_duration < 30 and speaking_metrics.get("total_words", 0) > 0:
                duration_adjustment = -15
                score += duration_adjustment
                breakdown["adjustments"].append({
                    "factor": "Duration",
                    "points": duration_adjustment,
                    "reason": f"Very short presentation ({total_duration:.0f}s)",
                    "evidence": f"{total_duration:.1f} seconds"
                })
            elif total_duration < 60:
                duration_adjustment = -8
                score += duration_adjustment
                breakdown["adjustments"].append({
                    "factor": "Duration",
                    "points": duration_adjustment,
                    "reason": f"Short presentation ({total_duration:.0f}s)",
                    "evidence": f"{total_duration:.1f} seconds"
                })
            
            # === FINAL CALCULATION ===
            # Cap score to valid range with validation
            final_score = max(0, min(100, round(score, 1)))
            
            # Determine category
            if final_score >= 86:
                category = "Excellent"
            elif final_score >= 71:
                category = "Good"
            elif final_score >= 51:
                category = "Average"
            elif final_score >= 31:
                category = "Below Average"
            else:
                category = "Needs Improvement"
            
            breakdown["final_score"] = final_score
            breakdown["category"] = category
            breakdown["raw_score"] = round(score, 1)
            
            logger.info(f"Voice confidence: {final_score}/100 ({category}) - Rate: {speaking_rate:.0f} WPM, Fillers: {filler_ratio*100:.1f}%, Pauses: {total_pause_count}")
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Error calculating voice confidence: {e}")
            return {
                "base_score": 50.0,
                "adjustments": [{"factor": "Error", "points": 0, "reason": str(e)}],
                "final_score": 50.0,
                "raw_score": 50.0,
                "category": "Unknown",
                "error": str(e)
            }

# Global enhanced audio analyzer instance
enhanced_audio_analyzer = EnhancedAudioAnalyzer()
