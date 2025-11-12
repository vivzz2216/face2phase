"""
Transcript processor for timestamped transcript generation
"""
import logging
from typing import Dict, List, Tuple, Optional
from models.enhanced_audio_analyzer import enhanced_audio_analyzer

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
            filler_events = self._collect_filler_events(audio_results.get('filler_analysis', {}))
            
            if words_with_timing and len(words_with_timing) > 0:
                words_source = words_with_timing
                if filler_events:
                    words_source = self._merge_words_and_fillers(words_with_timing, filler_events)
                return self._generate_from_word_timing(words_source)
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
        tolerance: float = 0.05
    ) -> List[Dict]:
        """Merge filler events into word timeline, inserting placeholders when needed."""
        merged = [dict(word) for word in words_with_timing]
        total_words = len(words_with_timing)
        
        for event in filler_events:
            word_index = event.get("word_index")
            if isinstance(word_index, int) and 0 <= word_index < total_words:
                # Word already represented in transcript timeline
                continue
            
            start = event.get("start")
            if start is None:
                continue
            end = event.get("end")
            if end is None:
                duration = event.get("duration")
                if isinstance(duration, (int, float)):
                    end = start + max(duration, 0.1)
                else:
                    end = start + 0.35
            
            token_original = (event.get("token_original") or event.get("label") or "").strip()
            if not token_original:
                token_original = "murmur"
            token_lower = token_original.lower().strip()
            if token_lower in {"murmur", "mumbling", "mumble"}:
                token_original = "mumbling"
            
            # Skip if a matching word already exists nearby
            duplicate = False
            for existing in words_with_timing:
                existing_start = existing.get("start")
                if existing_start is None:
                    continue
                if abs(existing_start - start) <= tolerance:
                    existing_word = (existing.get("word") or "").strip().lower().strip('.,!?;:')
                    if existing_word == token_original.lower().strip('.,!?;:'):
                        duplicate = True
                        break
            if duplicate:
                continue
            
            merged.append({
                "word": token_original,
                "start": float(start),
                "end": float(end),
                "is_filler": True,
                "confidence": float(event.get("confidence") or event.get("score") or 0.0),
                "method": event.get("method", "filler")
            })
        
        merged.sort(key=lambda word: word.get("start", 0.0))
        return merged
    
    def _generate_from_word_timing(self, words_with_timing: List[Dict]) -> List[Dict]:
        """Generate segments from word-level timing data"""
        segments = []
        current_segment = {"words": [], "start": None, "end": None}
        
        # Group words into sentences/phrases (every ~5-10 words or at punctuation)
        for word_data in words_with_timing:
            word = word_data.get('word', '')
            start = word_data.get('start', 0.0)
            end = word_data.get('end', 0.0)
            
            if current_segment["start"] is None:
                current_segment["start"] = start
            
            current_segment["words"].append(word)
            current_segment["end"] = end
            
            # Create segment every ~8 words or at sentence endings
            if (len(current_segment["words"]) >= 8 and 
                word.strip().endswith(('.', '!', '?'))) or len(current_segment["words"]) >= 12:
                
                text = ' '.join(current_segment["words"])
                segments.append({
                    "timestamp": self._format_timestamp(current_segment["start"]),
                    "text": text,
                    "start": current_segment["start"],
                    "end": current_segment["end"]
                })
                
                current_segment = {"words": [], "start": None, "end": None}
        
        # Add remaining words
        if current_segment["words"]:
            text = ' '.join(current_segment["words"])
            segments.append({
                "timestamp": self._format_timestamp(current_segment["start"]),
                "text": text,
                "start": current_segment["start"],
                "end": current_segment["end"]
            })
        
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





