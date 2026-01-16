"""
Composite scoring engine for presentation analytics.
Produces weighted sub-scores and badges using audio, visual, and text metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


@dataclass
class Scorecard:
    voice_delivery: float
    visual_presence: float
    narrative_clarity: float
    engagement: float
    composite: float
    badge: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_scores": {
                "voice_delivery": round(self.voice_delivery, 1),
                "visual_presence": round(self.visual_presence, 1),
                "narrative_clarity": round(self.narrative_clarity, 1),
                "engagement": round(self.engagement, 1)
            },
            "composite": round(self.composite, 1),
            "badge": self.badge,
            "details": self.details
        }


class ScoringEngine:
    VOICE_WEIGHT = 0.32
    VISUAL_WEIGHT = 0.28
    NARRATIVE_WEIGHT = 0.26
    ENGAGEMENT_WEIGHT = 0.14

    # Post-calibration parameters to improve separation
    CALIBRATION_CONTRAST = 1.22  # >1 increases spread
    CALIBRATION_CENTER = 57.0    # center point for contrast curve
    CALIBRATION_BASE_SHIFT = 3.0 # lift overall scores slightly

    BADGE_THRESHOLDS = [
        (90, "diamond"),
        (80, "platinum"),
        (70, "gold"),
        (60, "silver"),
        (50, "bronze"),
    ]

    def evaluate(
        self,
        audio_results: Dict[str, Any],
        facial_results: Dict[str, Any],
        text_results: Dict[str, Any],
        file_type: str = "video"
    ) -> Dict[str, Any]:
        voice_delivery = self._score_voice(audio_results)
        visual_presence = self._score_visual(facial_results, file_type=file_type)
        narrative_clarity = self._score_narrative(text_results)
        engagement = self._score_engagement(audio_results, facial_results)
        advanced = text_results.get("advanced_text_metrics") or {}
        tension_summary = (facial_results.get("tension_summary") or {})
        speaking_metrics = audio_results.get("speaking_metrics") or {}

        # Dynamic weight handling to avoid bias for audio-only sessions
        if file_type == "audio":
            w_voice = self.VOICE_WEIGHT
            w_visual = 0.0
            w_narr = self.NARRATIVE_WEIGHT
            w_eng = self.ENGAGEMENT_WEIGHT
        else:
            w_voice = self.VOICE_WEIGHT
            w_visual = self.VISUAL_WEIGHT
            w_narr = self.NARRATIVE_WEIGHT
            w_eng = self.ENGAGEMENT_WEIGHT

        weight_sum = max(1e-6, (w_voice + w_visual + w_narr + w_eng))
        composite = (
            voice_delivery * w_voice +
            visual_presence * w_visual +
            narrative_clarity * w_narr +
            engagement * w_eng
        ) / weight_sum

        # Apply post calibration to enhance separation using key indicators
        composite = self._post_calibrate(
            composite=composite,
            voice_delivery=voice_delivery,
            visual_presence=visual_presence,
            narrative_clarity=narrative_clarity,
            engagement=engagement,
            audio_results=audio_results,
            text_results=text_results,
        )

        badge = self._assign_badge(composite)

        details = {
            "filler_ratio": audio_results.get("filler_analysis", {}).get("filler_ratio"),
            "topic_coherence": advanced.get("topic_coherence_score"),
            "tension_percentage": tension_summary.get("tension_percentage"),
            "opening_confidence": (audio_results.get("advanced_audio_metrics") or {})
                .get("opening_confidence", {})
                .get("opening_confidence"),
            "conciseness_score": speaking_metrics.get("conciseness_score"),
            "avg_eye_contact_pct": tension_summary.get("avg_eye_contact_pct"),
            "distinct_2": advanced.get("distinct_2"),
            "compression_ratio": advanced.get("compression_ratio"),
        }

        scorecard = Scorecard(
            voice_delivery=voice_delivery,
            visual_presence=visual_presence,
            narrative_clarity=narrative_clarity,
            engagement=engagement,
            composite=composite,
            badge=badge,
            details=details
        )
        return scorecard.to_dict()

    def _score_voice(self, audio_results: Dict[str, Any]) -> float:
        base_score = audio_results.get("voice_confidence_score", 50.0)
        filler_ratio = audio_results.get("filler_analysis", {}).get("filler_ratio", 0.0)
        speaking_rate = audio_results.get("speaking_metrics", {}).get("speaking_rate_wpm", 0.0)
        conciseness_score = audio_results.get("speaking_metrics", {}).get("conciseness_score")

        filler_penalty = 0
        if filler_ratio > 0.08:
            filler_penalty = 18
        elif filler_ratio > 0.05:
            filler_penalty = 12
        elif filler_ratio > 0.03:
            filler_penalty = 6
        elif filler_ratio > 0.015:
            filler_penalty = 3

        pace_adjustment = 0.0
        if 130 <= speaking_rate <= 160:
            pace_adjustment = 8.0
        elif 120 <= speaking_rate < 130 or 160 < speaking_rate <= 170:
            pace_adjustment = 5.0
        elif 105 <= speaking_rate < 120 or 170 < speaking_rate <= 185:
            pace_adjustment = 1.5
        elif 95 <= speaking_rate < 105 or 185 < speaking_rate <= 200:
            pace_adjustment = -4.0
        elif speaking_rate > 0:
            pace_adjustment = -9.0

        if isinstance(conciseness_score, (int, float)):
            pace_adjustment += (conciseness_score - 75.0) * 0.08

        scored = base_score - filler_penalty + pace_adjustment
        return _clamp(scored)

    def _score_visual(self, facial_results: Dict[str, Any], file_type: str) -> float:
        if file_type == "audio":
            # No visual signal available; do not artificially boost or penalize
            return 0.0

        base_score = facial_results.get("facial_confidence_score", 55.0)
        tension_summary = facial_results.get("tension_summary") or {}
        tension_percentage = tension_summary.get("tension_percentage")
        eye_contact_stability = tension_summary.get("eye_contact_stability")
        avg_eye_contact_pct = tension_summary.get("avg_eye_contact_pct")
        if avg_eye_contact_pct is None:
            raw_eye_contact = facial_results.get("avg_eye_contact")
            if isinstance(raw_eye_contact, (int, float)):
                avg_eye_contact_pct = raw_eye_contact * 100

        if isinstance(tension_percentage, (int, float)):
            if tension_percentage > 40:
                base_score -= 12
            elif tension_percentage > 25:
                base_score -= 8
            elif tension_percentage < 10:
                base_score += 4

        if isinstance(eye_contact_stability, (int, float)):
            if eye_contact_stability >= 75:
                base_score += 6
            elif eye_contact_stability < 45:
                base_score -= 6

        if isinstance(avg_eye_contact_pct, (int, float)):
            if 55 <= avg_eye_contact_pct <= 75:
                base_score += 8
            elif 45 <= avg_eye_contact_pct < 55 or 75 < avg_eye_contact_pct <= 85:
                base_score += 3
            elif avg_eye_contact_pct < 40:
                base_score -= 10
            elif avg_eye_contact_pct > 90:
                base_score -= 4

        return _clamp(base_score)

    def _score_narrative(self, text_results: Dict[str, Any]) -> float:
        vocab_score = text_results.get("vocabulary_score", 55.0)
        advanced = text_results.get("advanced_text_metrics") or {}
        coherence = advanced.get("topic_coherence_score")
        keyword_density = (advanced.get("keyword_coverage") or {}).get("keyword_density")
        sentence_score = advanced.get("sentence_pattern_score")
        compression_ratio = advanced.get("compression_ratio")
        distinct_2 = advanced.get("distinct_2")
        repeated_pct = advanced.get("repeated_ngram_pct")
        subject_diversity = advanced.get("subject_verb_diversity")

        adjustments = 0.0
        if isinstance(coherence, (int, float)):
            adjustments += (coherence - 60) * 0.2
        if isinstance(keyword_density, (int, float)):
            if keyword_density > 12:
                adjustments += 4
            elif keyword_density < 4:
                adjustments -= 4
        if isinstance(sentence_score, (int, float)):
            adjustments += (sentence_score - 55) * 0.15
        if isinstance(compression_ratio, (int, float)):
            if compression_ratio <= 0.5:
                adjustments += 3
            elif compression_ratio >= 0.6:
                adjustments -= 4
        if isinstance(distinct_2, (int, float)):
            if distinct_2 >= 0.27:
                adjustments += 3
            elif distinct_2 < 0.18:
                adjustments -= 3
        if isinstance(repeated_pct, (int, float)):
            if repeated_pct > 35:
                adjustments -= 6
            elif repeated_pct > 20:
                adjustments -= 3
        if isinstance(subject_diversity, (int, float)):
            if subject_diversity >= 0.35:
                adjustments += 3
            elif subject_diversity < 0.2:
                adjustments -= 4

        return _clamp(vocab_score + adjustments)

    def _score_engagement(self, audio_results: Dict[str, Any], facial_results: Dict[str, Any]) -> float:
        opening = (audio_results.get("advanced_audio_metrics") or {}).get("opening_confidence", {})
        opening_score = opening.get("opening_confidence")
        voice_confidence = audio_results.get("voice_confidence_score", 50.0)
        emotion_distribution = facial_results.get("emotion_distribution") or {}
        positive_emotions = emotion_distribution.get("happy", 0) + emotion_distribution.get("neutral", 0)

        score = voice_confidence * 0.4
        if isinstance(opening_score, (int, float)):
            score += opening_score * 0.35
        score += positive_emotions * 100 * 0.2

        filler_ratio = audio_results.get("filler_analysis", {}).get("filler_ratio", 0)
        if filler_ratio > 0.06:
            score -= 6

        return _clamp(score)

    def _post_calibrate(
        self,
        composite: float,
        voice_delivery: float,
        visual_presence: float,
        narrative_clarity: float,
        engagement: float,
        audio_results: Dict[str, Any],
        text_results: Dict[str, Any],
    ) -> float:
        """
        Improve practical separation between weak/strong sessions by:
        - applying a gentle contrast curve around a center point
        - nudging based on strong/weak signal heuristics
        """
        # Contrast curve
        contrasted = (composite - self.CALIBRATION_CENTER) * self.CALIBRATION_CONTRAST + self.CALIBRATION_CENTER
        contrasted += self.CALIBRATION_BASE_SHIFT

        # Heuristic nudges
        nudge = 0.0
        filler_ratio = (audio_results.get("filler_analysis") or {}).get("filler_ratio")
        coherence = (text_results.get("advanced_text_metrics") or {}).get("topic_coherence_score")

        strong_indicators = 0
        if voice_delivery >= 72:
            strong_indicators += 1
        if narrative_clarity >= 70:
            strong_indicators += 1
        if engagement >= 68:
            strong_indicators += 1

        weak_indicators = 0
        if isinstance(filler_ratio, (int, float)) and filler_ratio > 0.06:
            weak_indicators += 1
        if isinstance(coherence, (int, float)) and coherence < 55:
            weak_indicators += 1

        # Apply nudges
        if strong_indicators >= 2:
            nudge += 3.0
        elif strong_indicators == 1:
            nudge += 1.5

        if weak_indicators >= 2:
            nudge -= 4.0
        elif weak_indicators == 1:
            nudge -= 2.0

        return _clamp(contrasted + nudge)

    def _assign_badge(self, composite_score: float) -> str:
        for threshold, badge in self.BADGE_THRESHOLDS:
            if composite_score >= threshold:
                return badge
        return "starter"


scoring_engine = ScoringEngine()

