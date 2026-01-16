"""
Transcript processor for timestamped transcript generation
"""
import logging
from typing import Dict, List, Tuple, Optional
try:
    from ..audio.enhanced_audio_analyzer import enhanced_audio_analyzer
except ImportError:
    from analysis.audio.enhanced_audio_analyzer import enhanced_audio_analyzer

logger = logging.getLogger(__name__)

class TranscriptProcessor:
    """Processes transcripts and generates timestamped versions"""
    
    def __init__(self):
        pass
    
    def get_timestamped_transcript(self, audio_results: Dict, transcript: str) -> List[Dict]:
        """
        Generate timestamped transcript segments
        
        Args:
            audio_results: Audio analysis results (may contain word-level timing)
            transcript: Full transcript text
            
        Returns:
            List of transcript segments with timestamps
        """
        try:
            # Try to get word-level timing from audio results
            words_with_timing = audio_results.get('words_with_timing', [])
            
            if words_with_timing and len(words_with_timing) > 0:
                # Use ONLY Whisper's direct transcription
                # Acoustic filler detection creates false positives on clean audio
                return self._generate_from_word_timing(words_with_timing)
            else:
                # Fallback: generate segments from transcript
                return self._generate_segments_from_text(transcript, audio_results)
                
        except Exception as e:
            logger.error(f"Error generating timestamped transcript: {e}")
            return [{"timestamp": "0:00", "text": transcript, "start": 0.0, "end": 0.0}]
    
    def _collect_filler_events(self, filler_analysis: Optional[Dict]) -> List[Dict]:
        """Collect filler/murmur events with timestamps."""
        if not filler_analysis:
            return []
        events: List[Dict] = []
        for source_key in ("text_model_fillers", "acoustic_events"):
            for event in filler_analysis.get(source_key, []) or []:
                if event is None:
                    continue
                start = event.get("start")
                if start is None:
                    continue
                events.append(dict(event))
        events.sort(key=lambda evt: evt.get("start", 0.0))
        return events
    
    def _merge_words_and_fillers(
        self,
        words_with_timing: List[Dict],
        filler_events: List[Dict],
        tolerance: float = 0.3  # Increased to catch more duplicates
    ) -> List[Dict]:
        """
        Merge acoustically detected filler events into word timeline.
        
        This injects detected filler sounds (uh, um, ah) that Whisper removed
        back into the accurate transcription at their correct positions.
        """
        merged = [dict(word) for word in words_with_timing]
        total_words = len(words_with_timing)
        
        # Map acoustic filler labels to common filler words
        filler_word_mapping = {
            "murmur": "uh",
            "mumbling": "um", 
            "mumble": "uh",
            "filler": "uh",
            "hesitation": "uh",
            "breath": "uh"
        }
        
        # Filter and deduplicate filler events before merging
        filtered_events = []
        MIN_CONFIDENCE = 0.3  # Minimum confidence for acoustic fillers
        MIN_DURATION = 0.05   # Minimum duration (50ms) - too short = noise
        
        for event in filler_events:
            # Skip if already in transcript
            word_index = event.get("word_index")
            if isinstance(word_index, int) and 0 <= word_index < total_words:
                continue
            
            start = event.get("start")
            if start is None:
                continue
            
            # Quality filters
            confidence = event.get("confidence") or event.get("score") or 0.0
            if confidence < MIN_CONFIDENCE:
                continue
                
            duration = event.get("duration") or 0.0
            if duration < MIN_DURATION:
                continue
            
            # Check for duplicates in already-added events
            is_duplicate = False
            for existing_event in filtered_events:
                existing_start = existing_event.get("start", 0.0)
                # If events are within 0.3 seconds, consider them duplicates
                if abs(existing_start - start) < tolerance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_events.append(event)
        
        # Now merge the filtered events
        for event in filtered_events:
            start = event.get("start")
            end = event.get("end")
            if end is None:
                duration = event.get("duration")
                if isinstance(duration, (int, float)):
                    end = start + max(duration, 0.1)
                else:
                    end = start + 0.35
            
            # Get the filler type and map to actual filler word
            token_original = (event.get("token_original") or event.get("label") or "").strip()
            if not token_original:
                token_original = "murmur"
                
            token_lower = token_original.lower().strip()
            
            # Map acoustic labels to actual filler words
            filler_word = filler_word_mapping.get(token_lower, token_lower)
            
            # If it's already a known filler word, keep it
            known_fillers = {"uh", "um", "ah", "er", "hmm", "oh", "eh", "erm"}
            if token_lower not in known_fillers:
                filler_word = "uh"  # Default to "uh" for unknown acoustic events
            else:
                filler_word = token_lower
            
            # Final check: Skip if any existing WORD is nearby (not just events)
            duplicate_with_word = False
            for existing in merged:
                existing_start = existing.get("start")
                if existing_start is None:
                    continue
                if abs(existing_start - start) <= 0.2:  # 200ms tolerance
                    duplicate_with_word = True
                    break
            
            if duplicate_with_word:
                continue
            
            # Add the filler word to the transcript
            merged.append({
                "word": filler_word,
                "start": float(start),
                "end": float(end),
                "is_filler": True,
                "confidence": float(event.get("confidence") or event.get("score") or 0.0),
                "method": event.get("method", "acoustic")
            })
        
        merged.sort(key=lambda word: word.get("start", 0.0))
        return merged
    
    def _generate_from_word_timing(self, words_with_timing: List[Dict]) -> List[Dict]:
        """
        Generate segments from word-level timing data
        
        Creates time-based segments (15-30 second intervals) with natural sentence breaks
        for better timestamp accuracy and readability.
        """
        if not words_with_timing:
            return []
        
        segments = []
        current_segment = {"words": [], "start": None, "end": None}
        
        # Target segment duration: shorter intervals for better sync
        TARGET_SEGMENT_DURATION = 15.0  # seconds (reduced from 20)
        MIN_SEGMENT_DURATION = 8.0     # seconds (reduced from 10)
        MAX_SEGMENT_DURATION = 25.0    # seconds (reduced from 40)
        
        for i, word_data in enumerate(words_with_timing):
            word = word_data.get('word', '')
            start = word_data.get('start', 0.0)
            end = word_data.get('end', 0.0)
            
            # Initialize segment start time
            if current_segment["start"] is None:
                current_segment["start"] = start
            
            current_segment["words"].append(word)
            current_segment["end"] = end
            
            # Calculate current segment duration
            segment_duration = end - current_segment["start"]
            
            # Check if we should end this segment
            should_break = False
            
            # Natural break points: sentence endings with punctuation
            is_sentence_end = word.strip().endswith(('.', '!', '?'))
            
            # Time-based breaking:
            # - If we've reached target duration and found a sentence end
            # - If we've exceeded max duration (force break)
            # - If this is the last word
            is_last_word = (i == len(words_with_timing) - 1)
            
            if segment_duration >= MIN_SEGMENT_DURATION:
                if is_sentence_end and segment_duration >= TARGET_SEGMENT_DURATION * 0.7:
                    # Natural break at sentence end near target time
                    should_break = True
                elif segment_duration >= MAX_SEGMENT_DURATION:
                    # Force break if too long
                    should_break = True
            
            # Always break at last word
            if is_last_word:
                should_break = True
            
            if should_break and current_segment["words"]:
                text = ' '.join(current_segment["words"])
                segments.append({
                    "timestamp": self._format_timestamp(current_segment["start"]),
                    "text": text,
                    "start": current_segment["start"],
                    "end": current_segment["end"]
                })
                
                # Reset for next segment
                current_segment = {"words": [], "start": None, "end": None}
        
        # Safety: If no segments were created (edge case), create one with all words
        if not segments and words_with_timing:
            all_text = ' '.join([w.get('word', '') for w in words_with_timing])
            segments = [{
                "timestamp": self._format_timestamp(words_with_timing[0].get('start', 0.0)),
                "text": all_text,
                "start": words_with_timing[0].get('start', 0.0),
                "end": words_with_timing[-1].get('end', 0.0)
            }]
        
        return segments
    
    def _generate_segments_from_text(self, transcript: str, audio_results: Dict) -> List[Dict]:
        """Generate segments by splitting text and estimating timestamps"""
        # Get duration from audio results
        duration = audio_results.get('speaking_metrics', {}).get('total_duration', 0)
        if duration == 0:
            # Estimate duration from speaking rate
            speaking_rate = audio_results.get('speaking_rate_wpm', 150)  # Default 150 WPM
            word_count = len(transcript.split())
            duration = (word_count / speaking_rate) * 60  # Convert to seconds
        
        # Split transcript into sentences
        sentences = self._split_into_sentences(transcript)
        
        if not sentences:
            return [{"timestamp": "0:00", "text": transcript, "start": 0.0, "end": duration}]
        
        # Calculate time per sentence
        words_per_sentence = [len(s.split()) for s in sentences]
        total_words = sum(words_per_sentence)
        
        segments = []
        current_time = 0.0
        
        for i, sentence in enumerate(sentences):
            if total_words > 0:
                sentence_duration = (len(sentence.split()) / total_words) * duration
            else:
                sentence_duration = duration / len(sentences)
            
            segments.append({
                "timestamp": self._format_timestamp(current_time),
                "text": sentence.strip(),
                "start": current_time,
                "end": current_time + sentence_duration
            })
            
            current_time += sentence_duration
        
        return segments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into MM:SS timestamp"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def get_transcript_text(self, audio_results: Dict) -> str:
        """
        Extract transcript text from audio results
        
        Args:
            audio_results: Audio analysis results
            
        Returns:
            Transcript text
        """
        # Try different possible keys
        transcript = (audio_results.get('transcript', '') or 
                     audio_results.get('transcription', {}).get('transcript', '') or
                     audio_results.get('enhanced_transcript', ''))
        
        return transcript if transcript else "No transcript available."

# Global instance
transcript_processor = TranscriptProcessor()





