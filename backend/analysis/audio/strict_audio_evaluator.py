"""
Lightweight strict audio evaluator stub to keep backend working.
Generates basic metrics and scores from words_with_timing and duration.
"""
from __future__ import annotations
from typing import List, Dict, Any


class StrictAudioEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, words_with_timing: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
        # Basic safety
        duration_sec = float(duration_sec or 0.0)
        num_words = max(0, int(len(words_with_timing or [])))
        unique_words = len({(w.get('word') or '').lower() for w in (words_with_timing or []) if w.get('word')})
        lexical_diversity_ttr = (unique_words / num_words) if num_words > 0 else 0.0

        # Words per minute
        wpm = (num_words / duration_sec * 60.0) if duration_sec > 0 else 0.0

        # Simple heuristic scoring
        # Target speaking pace ~ 110-160 wpm. Score decreases if too slow/fast.
        def pace_score(wpm_val: float) -> float:
            if wpm_val <= 0:
                return 0.0
            if 110 <= wpm_val <= 160:
                return 20.0  # max for pace
            # linear falloff outside range, capped at 0
            if wpm_val < 110:
                diff = 110 - wpm_val
                return max(0.0, 20.0 - diff * 0.25)
            diff = wpm_val - 160
            return max(0.0, 20.0 - diff * 0.25)

        # Clarity/pronunciation proxy: higher lexical diversity modestly increases clarity up to a cap
        clarity_score = max(0.0, min(25.0, 25.0 * lexical_diversity_ttr))

        # Coherence/grammar proxy: based on words and duration (avoid extremely short content)
        density = (num_words / duration_sec) if duration_sec > 0 else 0.0
        coherence_score = max(0.0, min(25.0, density * 2.0))  # cap at 25

        # Content accuracy proxy: proportional to lexical diversity and total words (diminishing returns)
        content_score = max(0.0, min(20.0, (lexical_diversity_ttr * 10.0) + min(10.0, num_words * 0.02)))

        fluency_score = pace_score(wpm)

        final_100 = min(100.0, clarity_score + fluency_score + coherence_score + content_score)

        # Flags and issues
        flags: List[str] = []
        issues: List[str] = []
        if duration_sec < 10:
            flags.append("very_short_audio")
            issues.append("Audio too short for reliable evaluation.")
        if wpm == 0:
            flags.append("no_speech_detected")
            issues.append("No speech detected or duration is zero.")
        elif wpm < 90:
            flags.append("pace_slow")
            issues.append("Speaking pace appears slow.")
        elif wpm > 180:
            flags.append("pace_fast")
            issues.append("Speaking pace appears fast.")
        if lexical_diversity_ttr < 0.3 and num_words >= 20:
            flags.append("low_lexical_diversity")
            issues.append("Vocabulary diversity seems low.")

        result: Dict[str, Any] = {
            "metrics": {
                "words": num_words,
                "unique_words": unique_words,
                "lexical_diversity_ttr": round(lexical_diversity_ttr, 3),
                "duration_sec": round(duration_sec, 2),
                "wpm": round(wpm, 1),
            },
            "scores": {
                "clarity_pronunciation_25": round(clarity_score, 2),
                "fluency_pace_20": round(fluency_score, 2),
                "coherence_grammar_25": round(coherence_score, 2),
                "content_accuracy_20": round(content_score, 2),
                "final_100": round(final_100, 2),
            },
            "flags": flags,
            "top_issues": issues[:5],
        }

        return result


