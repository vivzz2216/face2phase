"""
Enhanced Audio Analysis with Precise Gap Detection, Pitch Tracking, and Filler Detection
"""
import whisper
import librosa
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import soundfile as sf
from config import FILLER_WORDS, LONG_PAUSE_THRESHOLD, SHORT_PAUSE_THRESHOLD, WHISPER_MODEL_SIZE

from models.disfluency_detector import disfluency_detector
from utils.device_detector import device_manager
from utils.report_utils import (
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
        self.vad = None
        self._setup_ffmpeg()
        self._load_whisper_model()
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
    
    def _load_whisper_model(self):
        """Load Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} on device: {self.device}")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=self.device)
            logger.info(f"Successfully loaded Whisper model: {WHISPER_MODEL_SIZE}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.whisper_model = None
    
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
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Calculate adaptive threshold if not provided
            if energy_threshold is None:
                energy_threshold = self.calculate_adaptive_threshold(y, sr)
            
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
            return pauses
            
        except Exception as e:
            logger.error(f"Error detecting pauses: {e}")
            return []
    
    def track_pitch_variations(self, audio_path: Path) -> Dict:
        """
        Track pitch variations to detect high/low voice changes
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with pitch tracking data
        """
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=16000)
            
            # Extract pitch using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                frame_length=2048
            )
            
            # Convert frames to time
            times = librosa.times_like(f0, sr=sr)
            
            # Remove unvoiced sections
            voiced_f0 = f0[voiced_flag]
            voiced_times = times[voiced_flag]
            
            if len(voiced_f0) == 0:
                return {"pitch_variations": [], "avg_pitch": 0, "pitch_range": 0}
            
            # Calculate statistics
            avg_pitch = np.nanmean(voiced_f0)
            std_pitch = np.nanstd(voiced_f0)
            
            # Classify pitch levels
            pitch_variations = []
            for i, (time, pitch) in enumerate(zip(voiced_times, voiced_f0)):
                if not np.isnan(pitch):
                    # Classify as high, normal, or low
                    if pitch > avg_pitch + std_pitch:
                        level = "HIGH"
                    elif pitch < avg_pitch - std_pitch:
                        level = "LOW"
                    else:
                        level = "NORMAL"
                    
                    pitch_variations.append({
                        "time": float(time),
                        "pitch": float(pitch),
                        "level": level
                    })
            
            logger.info(f"Tracked {len(pitch_variations)} pitch points")
            
            return {
                "pitch_variations": pitch_variations,
                "avg_pitch": float(avg_pitch),
                "pitch_range": float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)),
                "pitch_std": float(std_pitch)
            }
            
        except Exception as e:
            logger.error(f"Error tracking pitch: {e}")
            return {"pitch_variations": [], "avg_pitch": 0, "pitch_range": 0}
    
    def transcribe_with_word_timestamps(self, audio_path: Path) -> Dict:
        """
        Transcribe audio with word-level timestamps using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detailed transcription with word-level timing
        """
        try:
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            logger.info(f"Starting detailed transcription: {audio_path}")
            
            # Load audio with librosa (Windows compatibility)
            audio_data, sr = librosa.load(str(audio_path), sr=16000)
            logger.info(f"Audio loaded: {len(audio_data)} samples, {sr} Hz")
            
            # Transcribe with word timestamps
            logger.info("Transcribing with word-level timestamps...")
            result = self.whisper_model.transcribe(
                audio_data,
                word_timestamps=True,
                verbose=False
            )
            
            transcript = result["text"].strip()
            segments = result.get("segments", [])
            
            logger.info(f"Transcription completed: {len(transcript)} characters, {len(segments)} segments")
            
            # Extract word-level timing
            words_with_timing = []
            for segment in segments:
                if "words" in segment:
                    for word_info in segment["words"]:
                        words_with_timing.append({
                            "word": word_info.get("word", "").strip(),
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                            "probability": word_info.get("probability", 0)
                        })
            
            logger.info(f"Extracted {len(words_with_timing)} words with timestamps")
            
            return {
                "transcript": transcript,
                "segments": segments,
                "words_with_timing": words_with_timing,
                "duration": result.get("duration", len(audio_data) / sr)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "transcript": "",
                "segments": [],
                "words_with_timing": [],
                "duration": 0
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
            audio_data, sr = librosa.load(str(audio_path), sr=16000)
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
                            
                            # Mumbling characteristics: high ZCR, low RMS, low spectral centroid
                            if zcr > 0.1 and rms < 0.03 and spectral_centroid < 2000:
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
                
                # Start new line with timestamp
                current_line = f"[{start:.1f}s]{gap_marker} "
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
            
            # 2. Detect precise pauses
            pauses = self.detect_precise_pauses(audio_path)
            
            # 3. Track pitch variations
            pitch_data = self.track_pitch_variations(audio_path)
            
            # Load audio once for duration calculations
            audio_signal, sr = librosa.load(str(audio_path), sr=16000)
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
            voice_confidence_score = self._calculate_voice_confidence(
                speaking_metrics,
                filler_analysis,
                pauses,
                pitch_data
            )
            
            filler_events = filler_analysis.get("acoustic_events", [])
            advanced_audio_metrics = {
                "filler_trend": compute_filler_trend(filler_events, total_duration),
                "pause_cadence": compute_pause_cadence(pauses),
                "opening_confidence": compute_opening_confidence(
                    filler_events,
                    pauses,
                    voice_confidence_score
                )
            }
            
            logger.info("Comprehensive audio analysis completed")
            
            return {
                "transcript": transcription_data["transcript"],
                "enhanced_transcript": enhanced_transcript,
                "pauses": pauses,
                "pause_summary": {
                    "total_pauses": len(pauses),
                    "long_pauses": len([p for p in pauses if p["type"] == "LONG_PAUSE"]),
                    "short_pauses": len([p for p in pauses if p["type"] == "SHORT_PAUSE"]),
                    "total_pause_time": sum(p["duration"] for p in pauses),
                    "longest_pause": max([p["duration"] for p in pauses]) if pauses else 0,
                    "avg_pause_duration": (sum(p["duration"] for p in pauses) / len(pauses)) if pauses else 0
                },
                "pitch_data": pitch_data,
                "filler_analysis": filler_analysis,
                "speaking_metrics": speaking_metrics,
                "voice_confidence_score": voice_confidence_score,
                "words_with_timing": transcription_data["words_with_timing"],
                "advanced_audio_metrics": advanced_audio_metrics,
                "analysis_successful": True
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive audio analysis: {e}")
            return {
                "transcript": "",
                "enhanced_transcript": "",
                "pauses": [],
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
        if audio_path and words_with_timing:
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
                            if float(metrics.get("confidence", 0.0)) < 0.55:
                                continue
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
                                if float(metrics.get("confidence", 0.0)) < 0.55:
                                    continue
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
        if audio_path:
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
            # Skip near-silent segments to avoid classifying short pauses as fillers
            if avg_rms < 0.012:
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
                high_zcr = avg_zcr > 0.12
                low_centroid = avg_centroid < 2200
                quiet = avg_rms < 0.06
                very_quiet = avg_rms < 0.03
                low_rolloff = avg_rolloff < 2800
                
                # Require at least two corroborating signals to reduce false positives
                if high_zcr and quiet:
                    is_filler = True
                elif low_centroid and very_quiet and low_rolloff:
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
                
                events.append({
                    "label": "murmur",
                    "start": float(start_time),
                    "end": float(end_time),
                    "duration": float(duration),
                    "method": "window_scan",
                    "confidence": float(metrics.get("confidence", 0.6)),
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
        """Calculate speaking metrics"""
        transcript = transcription_data.get("transcript", "")
        words = transcript.split()
        total_words = len(words)
        
        # Calculate actual speaking time (excluding pauses)
        pause_time = sum(p["duration"] for p in pauses)
        speaking_time = total_duration - pause_time
        
        # Calculate speaking rate
        speaking_rate = (total_words / speaking_time * 60) if speaking_time > 0 else 0
        conciseness_score = self._score_conciseness_from_rate(speaking_rate)
        
        return {
            "total_words": total_words,
            "total_duration": total_duration,
            "speaking_time": speaking_time,
            "pause_time": pause_time,
            "speaking_rate_wpm": speaking_rate,
            "pause_percentage": (pause_time / total_duration * 100) if total_duration > 0 else 0,
            "conciseness_score": conciseness_score
        }
    
    def _calculate_voice_confidence(self, speaking_metrics: Dict, filler_analysis: Dict, pauses: List[Dict], pitch_data: Dict) -> float:
        """
        Calculate voice confidence score - STRICT REALISTIC ALGORITHM
        Based on research: Perfect speech quality is rare (70-85 is excellent, 85+ is exceptional)
        """
        try:
            # Start with moderate base score (50 = average)
            score = 50.0
            
            # === SPEAKING RATE (0-25 points) ===
            speaking_rate = speaking_metrics.get("speaking_rate_wpm", 0)
            if 135 <= speaking_rate <= 160:
                score += 25  # Excellent range (professional presentations)
            elif 120 <= speaking_rate < 135 or 160 < speaking_rate <= 175:
                score += 18  # Good range
            elif 105 <= speaking_rate < 120 or 175 < speaking_rate <= 190:
                score += 12  # Acceptable
            elif 90 <= speaking_rate < 105 or 190 < speaking_rate <= 210:
                score += 5   # Below average
            elif speaking_rate < 90 or speaking_rate > 210:
                score -= 12  # Too slow or too fast
            
            # === FILLER WORD USAGE (0-20 points) ===
            filler_ratio = filler_analysis.get("filler_ratio", 0)
            if filler_ratio < 0.005:
                score += 20  # Excellent - almost no fillers
            elif filler_ratio < 0.01:
                score += 15  # Very good
            elif filler_ratio < 0.02:
                score += 10  # Good
            elif filler_ratio < 0.04:
                score += 5   # Acceptable
            elif filler_ratio < 0.06:
                score += 0   # Noticeable but not excessive
            elif filler_ratio < 0.10:
                score -= 10  # Too many fillers
            else:
                score -= 20  # Excessive filler usage
            
            # === PAUSE ANALYSIS (0-20 points) ===
            long_pauses = len([p for p in pauses if p["type"] == "LONG_PAUSE"])
            total_pauses = len(pauses)
            pause_percentage = speaking_metrics.get("pause_percentage", 0)
            
            # Long pause penalty
            if long_pauses == 0:
                score += 8  # No problematic pauses
            elif long_pauses <= 1:
                score += 5
            elif long_pauses <= 3:
                score += 2
            elif long_pauses <= 5:
                score -= 5
            else:
                score -= 15  # Too many long pauses
            
            # Natural pausing (good rhythm)
            if 5 <= total_pauses <= 15:
                score += 8  # Natural rhythm
            elif 3 <= total_pauses < 5 or 15 < total_pauses <= 20:
                score += 4  # Reasonable rhythm
            elif total_pauses < 3:
                score -= 5  # Too few pauses (rushed)
            elif total_pauses > 20:
                score -= 10  # Too many pauses (choppy)
            
            # Pause percentage penalty
            if pause_percentage > 30:
                score -= 10  # Too much silence
            elif pause_percentage > 25:
                score -= 5
            
            # === PITCH STABILITY (0-15 points) ===
            pitch_variance = pitch_data.get("pitch_variance", 0)
            if pitch_variance < 0.1:
                score += 15  # Very stable
            elif pitch_variance < 0.15:
                score += 12  # Stable
            elif pitch_variance < 0.20:
                score += 8   # Reasonably stable
            elif pitch_variance < 0.30:
                score += 3   # Somewhat unstable
            elif pitch_variance < 0.40:
                score -= 5   # Unstable
            else:
                score -= 15  # Very unstable
            
            # === ADDITIONAL FACTORS ===
            
            # Voice breaks (consistency)
            voice_breaks = speaking_metrics.get("voice_breaks_ratio", 0)
            if voice_breaks > 0.15:
                score -= 10  # Too many breaks
            elif voice_breaks > 0.20:
                score -= 15  # Many breaks
            
            # Stammering/Stuttering detection from filler_analysis
            if filler_analysis.get("stammering_detected", False):
                stutter_count = filler_analysis.get("stutter_count", 0)
                stutter_ratio = filler_analysis.get("stutter_ratio", 0)
                
                if stutter_count > 10:
                    score -= 25  # Severe stuttering
                elif stutter_count > 5:
                    score -= 15  # Moderate stuttering
                elif stutter_count > 2:
                    score -= 8   # Mild stuttering
                
                if stutter_ratio > 0.10:
                    score -= 20  # High stutter ratio
                elif stutter_ratio > 0.05:
                    score -= 10  # Significant stutter ratio
            
            # Overall speaking time (very short presentations score lower)
            total_duration = speaking_metrics.get("total_duration", 0)
            if total_duration < 30:  # Less than 30 seconds
                score -= 20  # Too short
            elif total_duration < 60:  # Less than 1 minute
                score -= 10  # Very short
            
            # === FINAL CALCULATION ===
            # Cap the score realistically: 85+ is exceptional, 100 is perfect
            final_score = max(25, min(95, score))  # 25-95 range (100 is unattainable in practice)
            
            logger.info(f"Voice confidence breakdown: base={50}, speaking_rate={speaking_rate}WPM, fillers={filler_ratio:.3f}, long_pauses={long_pauses}, pause_%={pause_percentage:.1f}%, pitch_var={pitch_variance:.3f}, final={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating voice confidence: {e}")
            return 50.0  # Return realistic average instead of 70

# Global enhanced audio analyzer instance
enhanced_audio_analyzer = EnhancedAudioAnalyzer()

