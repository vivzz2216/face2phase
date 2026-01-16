"""
Audio analysis module for voice confidence and speech clarity
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
    if hasattr(numba, 'jit'):
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
from pathlib import Path
import soundfile as sf
from ...core.settings import FILLER_WORDS, LONG_PAUSE_THRESHOLD, SHORT_PAUSE_THRESHOLD, WHISPER_MODEL_SIZE
from ...utils.device_detector import device_manager

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Analyzes audio for voice confidence and speech clarity"""
    
    def __init__(self):
        self.device = device_manager.get_device()
        self.whisper_model = None
        self._whisper_loaded = False  # Track if we've attempted to load
        
        # Set up FFmpeg path for Windows compatibility
        self._setup_ffmpeg()
        # LAZY LOADING: Don't load Whisper at startup
        # self._load_whisper_model()  # Commented out for fast startup
        logger.info("AudioAnalyzer created (Whisper will load on first use)")
    
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
            logger.warning("Whisper library not installed; audio transcription will be disabled.")
            self.whisper_model = None
            return False
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} on device: {self.device}")
            self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=self.device)
            logger.info(f"Successfully loaded Whisper model: {WHISPER_MODEL_SIZE}")
            logger.info(f"Model device: {next(self.whisper_model.parameters()).device}")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            logger.error(f"Device: {self.device}")
            self.whisper_model = None
            return False
    
    def extract_audio_features(self, audio_path: Path) -> Dict:
        """
        Extract audio features for voice confidence analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing audio features
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract fundamental frequency (pitch) with fallback for numba issues
            try:
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
                )
            except Exception as pyin_error:
                # Fallback to piptrack if pyin fails due to numba compatibility
                logger.warning(f"librosa.pyin failed (numba issue), using piptrack fallback: {pyin_error}")
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                # Extract pitch values
                f0_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch_val = pitches[index, t]
                    if pitch_val > 0:
                        f0_values.append(pitch_val)
                if f0_values:
                    f0 = np.array(f0_values)
                    voiced_flag = np.ones(len(f0_values), dtype=bool)
                else:
                    # No pitch found, use empty arrays
                    f0 = np.array([])
                    voiced_flag = np.array([], dtype=bool)
                voiced_probs = None
            
            # Extract energy
            energy = librosa.feature.rms(y=y)[0]
            
            # Extract zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Calculate statistics (handle empty arrays)
            if len(f0) > 0 and len(voiced_flag) > 0 and np.any(voiced_flag):
                voiced_pitches = f0[voiced_flag]
                pitch_mean = np.nanmean(voiced_pitches) if len(voiced_pitches) > 0 else 0.0
                pitch_std = np.nanstd(voiced_pitches) if len(voiced_pitches) > 0 else 0.0
            else:
                pitch_mean = 0.0
                pitch_std = 0.0
            energy_mean = np.mean(energy)
            energy_std = np.std(energy)
            zcr_mean = np.mean(zcr)
            
            # Detect voice breaks (low energy periods)
            energy_threshold = energy_mean * 0.3
            voice_breaks = np.sum(energy < energy_threshold) / len(energy)
            
            # Detect long pauses
            frame_length = 2048
            hop_length = 512
            frame_duration = hop_length / sr
            
            # Find silent frames
            silent_frames = energy < energy_threshold
            pause_lengths = []
            current_pause = 0
            
            for is_silent in silent_frames:
                if is_silent:
                    current_pause += frame_duration
                else:
                    if current_pause > 0:
                        pause_lengths.append(current_pause)
                        current_pause = 0
            
            # Count long pauses
            long_pauses = sum(1 for pause in pause_lengths if pause > LONG_PAUSE_THRESHOLD)
            short_pauses = sum(1 for pause in pause_lengths if pause > SHORT_PAUSE_THRESHOLD)
            
            return {
                "pitch_mean": float(pitch_mean) if not np.isnan(pitch_mean) else 0,
                "pitch_std": float(pitch_std) if not np.isnan(pitch_std) else 0,
                "energy_mean": float(energy_mean),
                "energy_std": float(energy_std),
                "zcr_mean": float(zcr_mean),
                "voice_breaks_ratio": float(voice_breaks),
                "long_pauses": int(long_pauses),
                "short_pauses": int(short_pauses),
                "total_pauses": len(pause_lengths),
                "duration": len(y) / sr
            }
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def transcribe_audio(self, audio_path: Path) -> Dict:
        """
        Transcribe audio using Whisper with Windows compatibility fixes
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary containing transcription and metadata
        """
        try:
            # LAZY LOADING: Load Whisper model on first use
            if not self.whisper_model:
                self._ensure_whisper_loaded()
            
            if not self.whisper_model:
                raise Exception("Whisper model not loaded")
            
            logger.info(f"Starting transcription of: {audio_path}")
            logger.info(f"Audio file exists: {audio_path.exists()}")
            logger.info(f"Audio file size: {audio_path.stat().st_size if audio_path.exists() else 'N/A'} bytes")
            
            # Windows compatibility fix: Load audio with librosa first
            try:
                import librosa
                logger.info("Loading audio with librosa for Windows compatibility...")
                
                # Load audio data directly
                audio_data, sample_rate = librosa.load(str(audio_path), sr=16000)
                logger.info(f"Audio loaded: {len(audio_data)} samples, {sample_rate} Hz")
                
                # Transcribe using audio data instead of file path
                logger.info("Transcribing audio data...")
                result = self.whisper_model.transcribe(audio_data, word_timestamps=False)
                
            except Exception as librosa_error:
                logger.warning(f"Librosa approach failed: {librosa_error}")
                logger.info("Falling back to file path approach...")
                
                # Fallback: Try file path with multiple formats
                audio_path_abs = audio_path.resolve()
                logger.info(f"Absolute audio path: {audio_path_abs}")
                
                # Check if file is accessible
                if not audio_path_abs.exists():
                    raise Exception(f"Audio file does not exist: {audio_path_abs}")
                
                # Try different path formats for Windows compatibility
                try:
                    result = self.whisper_model.transcribe(str(audio_path_abs), word_timestamps=False)
                except Exception as e1:
                    logger.warning(f"First attempt failed: {e1}")
                    # Try with forward slashes
                    try:
                        path_str = str(audio_path_abs).replace('\\', '/')
                        logger.info(f"Trying with forward slashes: {path_str}")
                        result = self.whisper_model.transcribe(path_str, word_timestamps=False)
                    except Exception as e2:
                        logger.warning(f"Second attempt failed: {e2}")
                        # Try copying to a simpler path
                        import tempfile
                        import shutil
                        temp_dir = Path(tempfile.gettempdir())
                        temp_audio = temp_dir / f"temp_audio_{audio_path_abs.stem}.wav"
                        logger.info(f"Copying to temp location: {temp_audio}")
                        shutil.copy2(audio_path_abs, temp_audio)
                        result = self.whisper_model.transcribe(str(temp_audio), word_timestamps=False)
                        # Clean up temp file
                        temp_audio.unlink()
            
            transcript = result["text"].strip()
            segments = result.get("segments", [])
            
            logger.info(f"Transcription completed: {len(transcript)} characters")
            logger.info(f"Number of segments: {len(segments)}")
            logger.info(f"Transcription preview: {transcript[:200]}...")
            
            # Calculate speaking rate
            total_words = len(transcript.split())
            duration = result.get("duration", 0)
            speaking_rate = (total_words / duration * 60) if duration > 0 else 0
            
            logger.info(f"Speaking rate: {speaking_rate:.1f} WPM")
            
            # Enhanced transcript with detailed analysis (include audio_path for voice quality)
            enhanced_transcript = self._create_enhanced_transcript(segments, transcript, audio_path)
            
            # Detect pauses and gaps
            pause_analysis = self._analyze_pauses(segments, duration)
            
            return {
                "transcript": transcript,
                "enhanced_transcript": enhanced_transcript,
                "segments": segments,
                "duration": duration,
                "word_count": total_words,
                "speaking_rate_wpm": speaking_rate,
                "pause_analysis": pause_analysis
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            logger.error(f"Audio path: {audio_path}")
            logger.error(f"Audio exists: {audio_path.exists() if audio_path else False}")
            return {"transcript": "", "segments": [], "duration": 0, "word_count": 0, "speaking_rate_wpm": 0}
    
    def _create_enhanced_transcript(self, segments: list, transcript: str, audio_path: Path = None) -> str:
        """
        Create an enhanced transcript with timing, pauses, voice quality, and stammering markers
        
        Args:
            segments: Whisper segments with timing information
            transcript: Raw transcript text
            audio_path: Path to audio file for voice quality analysis
            
        Returns:
            Enhanced transcript with detailed annotations
        """
        if not segments:
            return transcript
        
        enhanced_lines = []
        filled_words = ["um", "uh", "ah", "eh", "er", "hmm", "well", "like", "you know"]
        
        # Analyze voice energy levels for each segment
        voice_qualities = []
        if audio_path:
            try:
                import librosa
                audio_data, sr = librosa.load(str(audio_path), sr=16000)
                
                for segment in segments:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    
                    # Extract audio segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    if start_sample < len(audio_data) and end_sample <= len(audio_data):
                        segment_audio = audio_data[start_sample:end_sample]
                        
                        # Calculate energy (RMS)
                        if len(segment_audio) > 0:
                            energy = np.mean(np.abs(segment_audio))
                            voice_qualities.append({
                                "start": start_time,
                                "energy": energy,
                                "quality": self._classify_voice_quality(energy)
                            })
                        else:
                            voice_qualities.append({"start": start_time, "energy": 0, "quality": "silence"})
                    else:
                        voice_qualities.append({"start": start_time, "energy": 0, "quality": "unknown"})
            except Exception as e:
                logger.warning(f"Could not analyze voice quality: {e}")
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            # Calculate pause before this segment
            prev_end = segments[i-1].get("end", 0) if i > 0 else 0
            pause_duration = start_time - prev_end
            
            # Check for filler words and stammering
            words = text.lower().split()
            has_filler = any(filler in words for filler in filled_words)
            
            # Detect stammering (repeated words)
            has_stammering = False
            for j in range(len(words) - 2):
                if words[j] == words[j+1] and words[j] == words[j+2]:
                    has_stammering = True
                    # Add stammering marker
                    repeated_word = words[j]
                    text = text.replace(repeated_word, f"({repeated_word}{repeated_word}...)", 1)
                    break
            
            # Build enhanced line
            time_marker = f"[{start_time:.1f}s]"
            
            # Add pause marker
            pause_marker = ""
            if pause_duration > LONG_PAUSE_THRESHOLD:
                pause_marker = f" <PAUSE_{pause_duration:.1f}s>"
            elif pause_duration > SHORT_PAUSE_THRESHOLD:
                pause_marker = f" <break_{pause_duration:.1f}s>"
            
            # Add voice quality marker
            voice_marker = ""
            if i < len(voice_qualities):
                quality = voice_qualities[i].get("quality", "normal")
                if quality != "normal":
                    voice_marker = f" [{quality.upper()}]"
            
            # Add stammering marker
            stammering_marker = " (stammered)" if has_stammering else ""
            
            # Add filler marker (more natural)
            filler_marker = " (uhm...)" if has_filler and not has_stammering else ""
            
            enhanced_line = f"{time_marker}{pause_marker}{voice_marker} {text}{filler_marker}{stammering_marker}"
            enhanced_lines.append(enhanced_line)
        
        return "\n".join(enhanced_lines)
    
    def _classify_voice_quality(self, energy: float, threshold_low: float = 0.01, threshold_high: float = 0.1) -> str:
        """
        Classify voice quality based on energy level
        
        Args:
            energy: RMS energy value
            threshold_low: Low voice threshold
            threshold_high: High voice threshold
            
        Returns:
            Voice quality: "low", "normal", or "high"
        """
        if energy < threshold_low:
            return "low"
        elif energy > threshold_high:
            return "high"
        else:
            return "normal"
    
    def _analyze_pauses(self, segments: list, duration: float) -> Dict:
        """
        Analyze pauses and gaps in speech
        
        Args:
            segments: Whisper segments with timing
            duration: Total audio duration
            
        Returns:
            Dictionary with pause analysis
        """
        if not segments:
            return {
                "total_pauses": 0,
                "long_pauses": 0,
                "short_pauses": 0,
                "pause_positions": [],
                "total_silence_time": 0
            }
        
        pause_positions = []
        long_pauses = 0
        short_pauses = 0
        total_silence = 0
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get("end", 0)
            current_start = segments[i].get("start", 0)
            pause_duration = current_start - prev_end
            
            if pause_duration > SHORT_PAUSE_THRESHOLD:
                pause_positions.append({
                    "start": prev_end,
                    "end": current_start,
                    "duration": pause_duration
                })
                total_silence += pause_duration
                
                if pause_duration > LONG_PAUSE_THRESHOLD:
                    long_pauses += 1
                else:
                    short_pauses += 1
        
        # Calculate first pause (if speaking doesn't start immediately)
        first_start = segments[0].get("start", 0)
        if first_start > 0.5:
            pause_positions.append({
                "start": 0,
                "end": first_start,
                "duration": first_start
            })
            total_silence += first_start
            if first_start > LONG_PAUSE_THRESHOLD:
                long_pauses += 1
            else:
                short_pauses += 1
        
        # Calculate last pause (if speaking doesn't end at audio end)
        last_end = segments[-1].get("end", 0)
        if last_end < duration - 0.5:
            pause_positions.append({
                "start": last_end,
                "end": duration,
                "duration": duration - last_end
            })
            total_silence += (duration - last_end)
            if (duration - last_end) > LONG_PAUSE_THRESHOLD:
                long_pauses += 1
            else:
                short_pauses += 1
        
        return {
            "total_pauses": len(pause_positions),
            "long_pauses": long_pauses,
            "short_pauses": short_pauses,
            "pause_positions": pause_positions,
            "total_silence_time": total_silence,
            "silence_percentage": (total_silence / duration * 100) if duration > 0 else 0
        }
    
    def analyze_speech_clarity(self, transcript: str) -> Dict:
        """
        Analyze speech clarity from transcript
        
        Args:
            transcript: Transcribed text
            
        Returns:
            Dictionary containing clarity metrics
        """
        try:
            words = transcript.lower().split()
            total_words = len(words)
            
            if total_words == 0:
                return {
                    "filler_word_count": 0,
                    "filler_word_ratio": 0,
                    "unique_words": 0,
                    "vocabulary_richness": 0,
                    "stammering_detected": False
                }
            
            # Count filler words
            filler_count = 0
            filler_positions = []
            
            for i, word in enumerate(words):
                # Clean word (remove punctuation)
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word in FILLER_WORDS:
                    filler_count += 1
                    filler_positions.append(i)
            
            # Calculate vocabulary richness
            unique_words = len(set(words))
            vocabulary_richness = unique_words / total_words if total_words > 0 else 0
            
            # Detect stammering (repeated words/phrases)
            stammering_detected = False
            for i in range(len(words) - 2):
                if words[i] == words[i+1] == words[i+2]:
                    stammering_detected = True
                    break
            
            return {
                "filler_word_count": filler_count,
                "filler_word_ratio": filler_count / total_words if total_words > 0 else 0,
                "unique_words": unique_words,
                "vocabulary_richness": vocabulary_richness,
                "stammering_detected": stammering_detected,
                "filler_positions": filler_positions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speech clarity: {e}")
            return {}
    
    def calculate_voice_confidence_score(self, audio_features: Dict, clarity_metrics: Dict, pause_analysis: Dict = None) -> float:
        """
        Calculate voice confidence score (0-100) - IMPROVED ALGORITHM
        
        Args:
            audio_features: Audio feature dictionary
            clarity_metrics: Speech clarity metrics
            pause_analysis: Pause analysis data
            
        Returns:
            Voice confidence score
        """
        try:
            # Start with a base score of 70 (more generous baseline)
            score = 70.0
            
            # === POSITIVE FACTORS (Add points) ===
            
            # Reward good speaking rate (120-180 WPM is ideal)
            speaking_rate = clarity_metrics.get("speaking_rate_wpm", 0)
            if 120 <= speaking_rate <= 180:
                score += 15  # Ideal range bonus
            elif 100 <= speaking_rate < 120 or 180 < speaking_rate <= 200:
                score += 5   # Acceptable range
            
            # Reward low filler word usage
            filler_ratio = clarity_metrics.get("filler_word_ratio", 0)
            if filler_ratio < 0.02:
                score += 10  # Excellent - almost no fillers
            elif filler_ratio < 0.05:
                score += 5   # Good - few fillers
            
            # Reward steady pitch (not too much variance)
            pitch_std = audio_features.get("pitch_std", 0)
            if pitch_std < 30:
                score += 10  # Very steady voice
            elif pitch_std < 50:
                score += 5   # Reasonably steady
            
            # Reward consistent energy
            voice_breaks = audio_features.get("voice_breaks_ratio", 0)
            if voice_breaks < 0.1:
                score += 10  # Very consistent
            elif voice_breaks < 0.2:
                score += 5   # Fairly consistent
            
            # === NEGATIVE FACTORS (Subtract points - LESS PUNISHING) ===
            
            # Moderate penalty for excessive fillers
            if filler_ratio > 0.08:
                score -= min(15, filler_ratio * 100)  # Reduced from 25 to 15
            
            # Light penalty for many long pauses
            long_pauses = audio_features.get("long_pauses", 0)
            if long_pauses > 3:
                score -= min(10, (long_pauses - 3) * 2)  # Only penalize if > 3 long pauses
            
            # Light penalty for excessive silence (only if very high)
            if pause_analysis:
                silence_percentage = pause_analysis.get("silence_percentage", 0)
                if silence_percentage > 30:  # Only penalize if > 30% silence
                    score -= min(10, (silence_percentage - 30) / 2)
            
            # Light penalty for voice breaks (only if excessive)
            if voice_breaks > 0.3:
                score -= min(10, voice_breaks * 20)  # Reduced penalty
            
            # Light penalty for stammering
            if clarity_metrics.get("stammering_detected", False):
                score -= 5  # Reduced from 10
            
            # Light penalty for very high pitch variance
            if pitch_std > 80:  # Only very unstable voices
                score -= min(10, (pitch_std - 80) / 10)
            
            return max(30, min(100, score))  # Minimum 30, maximum 100
            
        except Exception as e:
            logger.error(f"Error calculating voice confidence score: {e}")
            return 65.0  # Default improved score
    
    def analyze_audio(self, audio_path: Path) -> Dict:
        """
        Complete audio analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Complete audio analysis results
        """
        try:
            logger.info(f"Starting audio analysis for: {audio_path}")
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio_path)
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Analyze speech clarity with enhanced transcript
            clarity_metrics = self.analyze_speech_clarity(transcription.get("transcript", ""))
            
            # Add enhanced transcript details to clarity metrics
            clarity_metrics["enhanced_transcript"] = transcription.get("enhanced_transcript", "")
            clarity_metrics["pause_analysis"] = transcription.get("pause_analysis", {})
            
            # Calculate voice confidence score (including pause analysis)
            voice_confidence = self.calculate_voice_confidence_score(
                audio_features, 
                clarity_metrics,
                transcription.get("pause_analysis", {})
            )
            
            # Combine all results
            results = {
                "audio_features": audio_features,
                "transcription": transcription,
                "clarity_metrics": clarity_metrics,
                "voice_confidence_score": voice_confidence,
                "analysis_successful": True
            }
            
            logger.info(f"Audio analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return {
                "audio_features": {},
                "transcription": {},
                "clarity_metrics": {},
                "voice_confidence_score": 0,
                "analysis_successful": False,
                "error": str(e)
            }

# Global audio analyzer instance
audio_analyzer = AudioAnalyzer()
