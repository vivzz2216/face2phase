"""
Wrapper around Hugging Face's silicone/disfluency-detection model to
identify filler words and disfluencies in transcripts, optionally aligning
them with word-level timestamps.
"""
from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = AutoModelForTokenClassification = pipeline = None

logger = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()


class DisfluencyDetector:
    """
    Detects textual fillers/disfluencies using the silicone/disfluency-detection model.
    """

    MODEL_NAME = "silicone/disfluency-detection"

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required for disfluency detection")

        if self._pipeline is None:
            with _INIT_LOCK:
                if self._pipeline is None:
                    logger.info("Loading disfluency detection model: %s", self.MODEL_NAME)
                    tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                    model = AutoModelForTokenClassification.from_pretrained(self.MODEL_NAME)
                    self._pipeline = pipeline(
                        "token-classification",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="simple"
                    )

    def detect(
        self,
        transcript: str,
        words_with_timing: Optional[List[Dict]] = None,
        segments: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Detect disfluencies/fillers in the transcript.

        Args:
            transcript: The transcript text.
            words_with_timing: Optional list of word dicts with 'word', 'start', 'end'.
            segments: Optional list of Whisper segments (each with 'text', 'start', 'end').

        Returns:
            Dict containing filler tokens, counts, and optional alignment data.
        """
        if not transcript or not transcript.strip():
            return {
                "fillers": [],
                "filler_counts": {},
                "filler_ratio": 0.0,
                "alignments": [],
            }

        try:
            self._ensure_pipeline()
            results = self._pipeline(transcript)
        except Exception as exc:  # pragma: no cover - HF pipeline failure
            logger.error("Disfluency detection failed: %s", exc)
            return {
                "fillers": [],
                "filler_counts": {},
                "filler_ratio": 0.0,
                "alignments": [],
                "error": str(exc),
            }

        fillers = []
        alignments = []
        for entry in results:
            if entry.get("entity_group") != "DISFLUENCY":
                continue
            token = entry.get("word", "").strip()
            if not token:
                continue
            fillers.append({
                "token": token,
                "score": float(entry.get("score", 0.0)),
                "char_start": int(entry.get("start", -1)) if entry.get("start") is not None else None,
                "char_end": int(entry.get("end", -1)) if entry.get("end") is not None else None,
            })

        filler_tokens = [f["token"] for f in fillers]
        counts = Counter(filler_tokens)
        total_words = len(transcript.split())
        filler_ratio = len(fillers) / total_words if total_words else 0.0

        if segments:
            alignments = self._align_to_segments(fillers, segments)
        elif words_with_timing:
            alignments = self._align_to_words(fillers, words_with_timing)

        return {
            "fillers": fillers,
            "filler_counts": dict(counts),
            "filler_ratio": filler_ratio,
            "alignments": alignments,
        }

    def _align_to_segments(self, fillers: List[Dict], segments: List[Dict]) -> List[Dict]:
        """Align detected fillers to Whisper segments."""
        alignments = []
        lower_segments = [
            {
                **seg,
                "text_lower": seg.get("text", "").lower(),
            }
            for seg in segments
        ]

        for filler in fillers:
            token = filler["token"].lower()
            matched = False
            for seg in lower_segments:
                if token in seg["text_lower"]:
                    alignments.append({
                        "token": token,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "segment_text": seg.get("text"),
                        "score": filler.get("score"),
                        "char_start": filler.get("char_start"),
                        "char_end": filler.get("char_end"),
                    })
                    matched = True
                    break
            if not matched:
                alignments.append({
                    "token": token,
                    "start": None,
                    "end": None,
                    "segment_text": None,
                    "score": filler.get("score"),
                    "char_start": filler.get("char_start"),
                    "char_end": filler.get("char_end"),
                })
        return alignments

    def _align_to_words(self, fillers: List[Dict], words_with_timing: List[Dict]) -> List[Dict]:
        """Align detected fillers to word-level timings."""
        alignments = []
        lower_words = [
            {
                "start": word.get("start"),
                "end": word.get("end"),
                "word": word.get("word", "").strip().lower(),
            }
            for word in words_with_timing
            if word.get("word")
        ]

        for idx, filler in enumerate(fillers):
            token = filler["token"].lower()
            matched = False
            for word_idx, word in enumerate(lower_words):
                normalized = word["word"].replace(" ", "")
                if token.replace(" ", "") == normalized:
                    alignments.append({
                        "token": token,
                        "start": word["start"],
                        "end": word["end"],
                        "score": filler.get("score"),
                        "word_index": word_idx,
                        "char_start": filler.get("char_start"),
                        "char_end": filler.get("char_end"),
                    })
                    matched = True
                    break
            if not matched:
                alignments.append({
                    "token": token,
                    "start": None,
                    "end": None,
                    "score": filler.get("score"),
                    "word_index": None,
                    "char_start": filler.get("char_start"),
                    "char_end": filler.get("char_end"),
                })
        return alignments


disfluency_detector = DisfluencyDetector()

