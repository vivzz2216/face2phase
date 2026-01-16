from collections import Counter
from typing import Dict, List

def compute_filler_trend(filler_events: List[Dict], total_duration: float) -> Dict:
    """
    Aggregate filler events into time buckets for trend visualization.
    Returns counts per minute and top labels.
    """
    bucket_size = 60  # seconds
    buckets = {}
    label_counts = Counter()
    for event in filler_events:
        start = float(event.get("start", 0))
        bucket = int(start // bucket_size)
        label = event.get("label") or event.get("token_original") or "filler"
        buckets.setdefault(bucket, Counter())
        buckets[bucket][label] += 1
        label_counts[label] += 1
    trend = []
    for bucket, counter in sorted(buckets.items()):
        trend.append({
            "bucket_start": bucket * bucket_size,
            "bucket_label": f"{bucket * bucket_size}-{(bucket + 1) * bucket_size}s",
            "counts": dict(counter)
        })
    return {
        "trend": trend,
        "top_labels": label_counts.most_common(5),
        "total_duration": total_duration
    }


def compute_pause_cadence(pauses: List[Dict]) -> Dict:
    """
    Classify pauses into short/medium/long buckets and produce aggregate stats.
    """
    # Ensure pauses is a valid list
    if not pauses or not isinstance(pauses, list):
        pauses = []
    
    buckets = {"short": 0, "medium": 0, "long": 0}
    bucket_durations = {"short": 0.0, "medium": 0.0, "long": 0.0}
    total_duration = 0.0

    for pause in pauses:
        # Validate pause structure
        if not isinstance(pause, dict):
            continue
            
        # Extract duration - handle multiple possible field names
        duration = pause.get("duration") or pause.get("pause_duration") or 0.0
        try:
            duration = float(duration)
            if duration <= 0 or duration != duration:  # Check for NaN or negative
                continue
        except (ValueError, TypeError):
            continue
        
        total_duration += duration
        if duration < 1.0:
            key = "short"
        elif duration < 2.5:
            key = "medium"
        else:
            key = "long"
        buckets[key] += 1
        bucket_durations[key] += duration

    avg_duration = total_duration / len(pauses) if pauses else 0.0
    return {
        "counts": buckets,
        "durations": {k: round(v, 2) for k, v in bucket_durations.items()},
        "average_duration": round(avg_duration, 2),
        "total_pause_time": round(total_duration, 2)
    }


def compute_opening_confidence(
    filler_events: List[Dict],
    pauses: List[Dict],
    base_voice_score: float
) -> Dict:
    """
    Estimate confidence in the opening 30 seconds using filler rate and pause cadence.
    """
    opening_window = 30.0
    opening_fillers = [f for f in filler_events if float(f.get("start", 0)) <= opening_window]
    opening_pauses = [p for p in pauses if float(p.get("start", 0)) <= opening_window]

    filler_penalty = min(len(opening_fillers) * 2, 30)
    pause_penalty = min(len(opening_pauses) * 1.5, 20)
    base_score = max(base_voice_score or 50, 30)

    opening_score = max(0, min(100, base_score - filler_penalty - pause_penalty))

    return {
        "opening_confidence": round(opening_score, 1),
        "opening_filler_count": len(opening_fillers),
        "opening_pause_count": len(opening_pauses),
        "estimated_penalties": {
            "filler": filler_penalty,
            "pause": pause_penalty
        }
    }


def compute_tension_metrics(facial_results: Dict) -> Dict:
    """
    Provide aggregate tension ratio from facial analysis.
    """
    tension_pct = facial_results.get("tension_percentage")
    if tension_pct is None:
        tension_moments = facial_results.get("tension_moments") or []
        total_frames = len(facial_results.get("frame_results", [])) or 1
        tension_pct = (len(tension_moments) / total_frames) * 100

    avg_eye_contact = facial_results.get("avg_eye_contact", 0) * 100
    stability = facial_results.get("eye_contact_timeline") or []
    stability_score = 0
    if stability:
        stability_score = sum(frame.get("eye_contact_score", 0) for frame in stability) / len(stability)
        stability_score = round(stability_score * 100, 1)

    return {
        "tension_percentage": round(tension_pct, 1),
        "avg_eye_contact_pct": round(avg_eye_contact, 1),
        "eye_contact_stability": stability_score
    }


def smooth_emotion_timeline(emotion_timeline: List[Dict], window: int = 3) -> List[Dict]:
    """
    Smooth emotion timeline via moving average on confidence.
    """
    if not emotion_timeline:
        return []
    smoothed = []
    buffer: List[Dict] = []

    for entry in emotion_timeline:
        buffer.append(entry)
        if len(buffer) > window:
            buffer.pop(0)
        confidence_avg = sum(item.get("confidence", 0) for item in buffer) / len(buffer)
        smoothed.append({
            "timestamp": entry.get("timestamp"),
            "dominant_emotion": entry.get("emotion"),
            "confidence": round(confidence_avg, 3)
        })

    return smoothed


def compute_topic_coherence(text_metrics: Dict) -> Dict:
    """
    Assemble topic coherence and keyword coverage.
    """
    coherence = text_metrics.get("topic_coherence", {})
    coverage = text_metrics.get("keyword_coverage", {})
    sentence_patterns = text_metrics.get("sentence_patterns", {})

    return {
        "topic_coherence_score": coherence.get("score"),
        "top_topics": coherence.get("topics", [])[:5],
        "keyword_coverage": coverage,
        "sentence_pattern_score": sentence_patterns.get("variety_score"),
        "repetition_alerts": sentence_patterns.get("repetitions", [])
    }

