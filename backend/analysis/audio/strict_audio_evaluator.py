"""
Lightweight strict audio evaluator stub to keep backend working.
Generates basic metrics and scores from words_with_timing and duration.
"""
from __future__ import annotations
from typing import List, Dict, Any


class StrictAudioEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, words_with_timing: List[Dict[str, Any]], duration_sec: float, filler_stats: Dict[str, Any] = None, pause_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        # Basic metrics
        duration_sec = float(duration_sec or 0.0)
        num_words = max(0, int(len(words_with_timing or [])))
        
        # Word analysis
        words_list = [(w.get('word') or '').strip().lower() for w in (words_with_timing or [])]
        words_list = [w for w in words_list if w] # Filter empty
        
        unique_words = len(set(words_list))
        lexical_diversity_ttr = (unique_words / num_words) if num_words > 0 else 0.0
        
        # Complex words (length > 4 chars)
        complex_words = len([w for w in words_list if len(w) > 4])
        
        # Sentence structure approximation (split by punctuation or gaps > 0.6s)
        # Since we might not have punctuation, we assume average sentence length of 10-15 words is good
        avg_sentence_length = 12.0
        
        # Words per minute
        wpm = (num_words / duration_sec * 60.0) if duration_sec > 0 else 0.0

        # --- SCORING LOGIC (MEDIUM STRICT) ---
        
        # Extract stats
        filler_count = (filler_stats or {}).get('total_fillers', 0)
        total_pauses = (pause_stats or {}).get('total_pauses', 0)
        if not total_pauses and pause_stats and 'pause_count' in pause_stats:
             total_pauses = pause_stats['pause_count']

        # 1. Fluency & Pace (25 pts)
        # Target: 130-170 WPM is ideal.
        # Penalties: -1 per 2 fillers, -1 per 3 pauses (if excessive)
        def calculate_pace_score(wpm_val: float, fillers: int, pauses: int) -> float:
            base_score = 0.0
            if wpm_val <= 0: return 0.0
            
            # Base WPM score
            if 130 <= wpm_val <= 170: base_score = 25.0
            elif 110 <= wpm_val < 130: base_score = 22.0
            elif 170 < wpm_val <= 190: base_score = 23.0
            elif wpm_val < 110: 
                diff = 110 - wpm_val
                base_score = max(5.0, 22.0 - (diff * 0.4))
            elif wpm_val > 190:
                diff = wpm_val - 190
                base_score = max(5.0, 23.0 - (diff * 0.3))
            
            # Apply Penalties
            # Filler penalty: -0.5 points per filler (medium strict)
            filler_penalty = fillers * 0.5
            
            # Pause penalty: -0.3 points per pause, but allow first 2 pauses free
            pause_penalty = max(0, (pauses - 2) * 0.3)
            
            final_score = base_score - filler_penalty - pause_penalty
            return max(5.0, final_score) # Don't go below 5

        fluency_score = calculate_pace_score(wpm, filler_count, total_pauses)

        # 2. Clarity & Pronunciation (25 pts)
        # Penalize for repetitive speech (stammering) and high filler usage affecting clarity
        clarity_base = 25.0
        
        if lexical_diversity_ttr < 0.4:
            clarity_base -= 5.0 # Repetitive speech
            
        # Heavy filler usage also impacts clarity perception
        if filler_count > 5:
             clarity_base -= 2.0
             
        clarity_score = clarity_base

        # 3. Coherence & Grammar (25 pts)
        # Approximation: Healthy mix of simple (short) and complex (long) words indicates grammar
        # Ideal: 20-40% complex words
        complex_ratio = complex_words / num_words if num_words > 0 else 0.0
        if 0.2 <= complex_ratio <= 0.5:
            coherence_score = 25.0
        elif 0.1 <= complex_ratio < 0.2:
            coherence_score = 20.0
        else:
            coherence_score = 15.0 # Too simple or too academic

        # 4. Content Accuracy (25 pts)
        # Rewards unique vocabulary and reasonable duration
        # Cap at 25 points
        vocab_component = min(18.0, unique_words * 0.36) # 50 unique words = 18 pts
        length_component = min(7.0, duration_sec * 0.28) # 25 seconds = 7 pts
        content_score = vocab_component + length_component

        # Calculate final score
        final_100 = min(100.0, clarity_score + fluency_score + coherence_score + content_score)

        # Flags and issues
        flags: List[str] = []
        issues: List[str] = []
        
        if duration_sec < 10:
            flags.append("very_short_audio")
            issues.append("Audio is very short, hard to analyze accurately.")
            final_100 *= 0.8 # Penalty for very short samples
            
        if wpm < 100:
            flags.append("pace_slow")
            issues.append(f"Speaking pace is slow ({int(wpm)} WPM). Aim for 130-150 WPM.")
            
        if lexical_diversity_ttr < 0.5 and num_words > 30:
            flags.append("repetitive_vocab")
            issues.append(f"Vocabulary is repetitive ({int(lexical_diversity_ttr*100)}% unique). Try variable word choice.")

        result: Dict[str, Any] = {
            "metrics": {
                "words": num_words,
                "unique_words": unique_words,
                "lexical_diversity_ttr": round(lexical_diversity_ttr, 3),
                "duration_sec": round(duration_sec, 2),
                "wpm": round(wpm, 1),
            },
            "scores": {
                "clarity_pronunciation_25": round(clarity_score, 1),
                "fluency_pace_25": round(fluency_score, 1),
                "coherence_grammar_25": round(coherence_score, 1),
                "content_accuracy_25": round(content_score, 1),
                "final_100": round(final_100, 1),
            },
            "flags": flags,
            "top_issues": issues[:5],
        }

        return result



