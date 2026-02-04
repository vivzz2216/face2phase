"""
Composite scoring engine for presentation analytics.
Produces weighted sub-scores and badges using audio, visual, and text metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


import math


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, float(value)))


# ---------- Research-Backed Scoring Functions ----------

def _pace_score_gaussian(wpm: float) -> float:
    """
    Gaussian scoring for speaking rate (Tauroza & Allison).
    Center (mu) = 155 WPM, Width (sigma) = 25.
    Returns +8.0 at peak, drops smoothly to negative.
    Normalized to match original scale: Peak +8, Neutral 0, Penalty negative.
    """
    if wpm <= 0: return -9.0
    mu = 155.0
    sigma = 25.0
    # Raw 0-1 scale
    gaussian = math.exp(-((wpm - mu) ** 2) / (2 * sigma ** 2))
    # Map to range [-9.0, +8.0] approximately
    # At peak (1.0) -> +8.0
    # At 2 sigma (0.13) -> ~ -4.0
    return (17.0 * gaussian) - 9.0

def _filler_penalty_logistic(filler_ratio: float) -> float:
    """
    Logistic penalty for filler words (De Jong & Bosker).
    Inflection at 5%, slope k=120.
    Returns negative penalty (e.g. -12 at 5% to match legacy tiers).
    """
    k = 120.0
    x0 = 0.05
    max_penalty = -20.0
    penalty = max_penalty / (1.0 + math.exp(-k * (filler_ratio - x0)))
    return penalty

def _eye_contact_score_smooth(avg_eye_frac: float) -> float:
    """
    Smooth piecewise function for eye contact.
    Avoids hard bins and penalizes staring (>90%).
    Returns adjustment value (e.g. +8, -10).
    """
    # Ideal range 0.55 - 0.75 -> +8
    if 0.55 <= avg_eye_frac <= 0.75:
        return 8.0
    
    # Near ideal -> +3
    if 0.45 <= avg_eye_frac < 0.55 or 0.75 < avg_eye_frac <= 0.85:
        return 3.0
        
    # Low eye contact -> steep drop to -10
    if avg_eye_frac < 0.40:
        return -10.0
        
    # Staring -> penalty
    if avg_eye_frac > 0.90:
        return -4.0
        
    # Fallback / Smooth Interpolation for gaps
    return _clamp((avg_eye_frac - 0.55) * 40.0, -10.0, 8.0)

def _tension_penalty_logistic(tension_frac: float) -> float:
    """
    Logistic penalty for facial tension.
    """
    if tension_frac <= 0.10:
        return 4.0 # Bonus for very relaxed
    # Logistic curve centered at 0.30
    return -12.0 / (1.0 + math.exp(-40.0 * (tension_frac - 0.30)))



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

        # 1. Smooth Logistic Filler Penalty
        filler_penalty = _filler_penalty_logistic(filler_ratio)

        # 2. Smooth Gaussian Pace Adjustment
        pace_adjustment = _pace_score_gaussian(speaking_rate)

        # 3. Conciseness Adjustment
        conciseness_adj = 0.0
        if isinstance(conciseness_score, (int, float)):
            conciseness_adj = (conciseness_score - 75.0) * 0.08

        # Combine
        scored = base_score + filler_penalty + pace_adjustment + conciseness_adj
        return _clamp(scored)

    def _score_visual(self, facial_results: Dict[str, Any], file_type: str) -> float:
        if file_type == "audio":
            return 0.0

        base_score = facial_results.get("facial_confidence_score", 55.0)
        tension_summary = facial_results.get("tension_summary") or {}
        tension_percentage = tension_summary.get("tension_percentage")
        eye_contact_stability = tension_summary.get("eye_contact_stability")
        avg_eye_contact_pct = tension_summary.get("avg_eye_contact_pct")
        
        # Normalize inputs
        if avg_eye_contact_pct is None:
            raw_eye_contact = facial_results.get("avg_eye_contact")
            if isinstance(raw_eye_contact, (int, float)):
                avg_eye_contact_pct = raw_eye_contact * 100
        
        avg_eye_frac = (avg_eye_contact_pct or 0.0) / 100.0
        tension_frac = (tension_percentage or 0.0) / 100.0
        stability_score = (eye_contact_stability or 50.0) / 100.0

        # Smart Adjustments
        tension_adj = _tension_penalty_logistic(tension_frac)
        eye_contact_adj = _eye_contact_score_smooth(avg_eye_frac)
        
        # Stability: Linear map (0.0 -> -5, 1.0 -> +6)
        stability_adj = _clamp(stability_score * 11.0 - 5.0, -5.0, 6.0)

        # Negative Emotion Penalty check
        emotion_dist = facial_results.get("emotion_distribution") or {}
        neg_emotion_frac = sum(emotion_dist.get(k, 0) for k in ['angry', 'sad', 'fear', 'disgust'])
        neg_penalty = -15.0 if neg_emotion_frac > 0.30 else 0.0

        scored = base_score + tension_adj + eye_contact_adj + stability_adj + neg_penalty
        return _clamp(scored)

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
        opening_score = opening.get("opening_confidence") or 50.0
        voice_confidence = audio_results.get("voice_confidence_score", 50.0)
        emotion_distribution = facial_results.get("emotion_distribution") or {}
        positive_emotions = emotion_distribution.get("happy", 0) + emotion_distribution.get("neutral", 0)

        # Weighted Mix
        score = (voice_confidence * 0.40) + (opening_score * 0.35) + (positive_emotions * 100 * 0.20)
        
        # Smooth Filler Penalty for Engagement
        filler_ratio = audio_results.get("filler_analysis", {}).get("filler_ratio", 0)
        if filler_ratio > 0.06:
            # Scaled penalty (e.g. -6 at 6%, -12 at 12%)
            score -= 6.0 * (filler_ratio / 0.06)

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
        Research-backed calibration (Kolen & Brennan).
        Reduced slope (1.18), continuous nudges, and safety clamps.
        """
        # 1. Calibrated Contrast
        # Slope 1.18 (gentle separation), Center shift +3.0
        contrasted = (composite - 57.0) * 1.18 + 60.0
        
        # 2. Continuous Nudge Module
        # Translate "strong indicators" to smooth bonus
        def norm(x, lo=40, hi=90):
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))

        # Strength bonus (0 to +3)
        strong_score = (norm(voice_delivery) + norm(narrative_clarity) + norm(engagement)) / 3.0
        bonus = 3.0 * strong_score

        # Weakness penalty (0 to -4)
        filler_ratio = (audio_results.get("filler_analysis") or {}).get("filler_ratio") or 0.0
        coherence = (text_results.get("advanced_text_metrics") or {}).get("topic_coherence_score") or 60.0
        
        weak_count = 0.0
        if filler_ratio > 0.06: weak_count += 1.0
        if coherence < 55: weak_count += 1.0
        
        penalty = -4.0 * (weak_count / 2.0)

        # 3. Final Clamp [35, 88] to prevent inflation/collapse
        final_score = contrasted + bonus + penalty
        # Allow exceptional scores >88 only if inputs are truly exceptional (>90)
        if composite > 90:
            return _clamp(final_score, 35.0, 98.0)
        return _clamp(final_score, 35.0, 88.0)

    def _assign_badge(self, composite_score: float) -> str:
        for threshold, badge in self.BADGE_THRESHOLDS:
            if composite_score >= threshold:
                return badge
        return "starter"


scoring_engine = ScoringEngine()

