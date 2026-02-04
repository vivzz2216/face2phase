"""
Sentence Opener Analysis Module

Detects and analyzes repetitive sentence-starting words/phrases
"""

import re
from typing import Dict, List, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Common sentence openers that should be tracked
COMMON_OPENERS = [
    "so", "and", "but", "well", "actually", "basically", "literally",
    "like", "you know", "i think", "i mean", "uhm", "uh", "ah",
    "okay", "right", "now", "then", "first", "second", "next"
]

# Suggested varied alternatives
OPENER_ALTERNATIVES = {
    "so": ["Therefore", "Consequently", "As a result", "Thus", "Hence", "Moving forward"],
    "and": ["Additionally", "Furthermore", "Moreover", "In addition", "Also", "Similarly"],
    "but": ["However", "Nevertheless", "Nonetheless", "On the other hand", "Conversely", "Yet"],
    "well": ["To begin with", "Firstly", "Initially", "To start", "Let me explain"],
    "now": ["At this point", "Currently", "Moving on", "Proceeding to", "Next"],
    "actually": ["In fact", "Indeed", "As a matter of fact", "Truthfully"],
    "basically": ["Essentially", "Fundamentally", "In essence", "Primarily"],
    "literally": ["Precisely", "Exactly", "Specifically", "Actually"],
    "okay": ["Alright", "Very well", "Understood", "Agreed"],
    "right": ["Correct", "Indeed", "Precisely", "Exactly"],
    "first": ["Firstly", "To begin with", "Initially", "First of all"],
    "i think": ["In my opinion", "I believe", "It seems to me", "From my perspective"],
    "i mean": ["That is to say", "In other words", "Specifically", "To clarify"],
    "you know": ["As you're aware", "Obviously", "Clearly", "Evidently"]
}

def analyze_sentence_openers(transcript: str, words_with_timing: List[Dict] = None) -> Dict:
    """
    Analyze sentence openers in the transcript
    
    Args:
        transcript: Full transcript text
        words_with_timing: Optional word-level timing data
        
    Returns:
        Dictionary with sentence opener analysis
    """
    if not transcript or not transcript.strip():
        return {
            "status": "no_data",
            "message": "No transcript available for analysis",
            "openers_found": {},
            "total_sentences": 0,
            "recommendations": []
        }
    
    # Split into sentences - handle multiple punctuation marks
    sentences = re.split(r'[.!?]+\s+', transcript)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return {
            "status": "no_data",
            "message": "No complete sentences found",
            "openers_found": {},
            "total_sentences": 0,
            "recommendations": []
        }
    
    # Extract first 1-3 words from each sentence (covers multi-word openers)
    sentence_openers = []
    opener_positions = []  # Track which sentences used which openers
    
    for idx, sentence in enumerate(sentences):
        words = sentence.lower().split()
        if not words:
            continue
        
        # Check single word opener
        first_word = words[0].strip('.,!?;:"\'')
        sentence_openers.append(first_word)
        opener_positions.append((idx, first_word))
        
        # Check two-word phrases
        if len(words) >= 2:
            two_word = f"{words[0]} {words[1]}".strip('.,!?;:"\'').lower()
            if two_word in COMMON_OPENERS:
                sentence_openers.append(two_word)
                opener_positions.append((idx, two_word))
    
    # Count openers
    opener_counts = Counter(sentence_openers)
    
    # Filter to only tracked openers
    tracked_openers = {
        opener: count 
        for opener, count in opener_counts.items() 
        if opener in COMMON_OPENERS
    }
    
    # Calculate percentages
    total_sentences = len(sentences)
    opener_analysis = {}
    
    for opener, count in tracked_openers.items():
        percentage = (count / total_sentences) * 100
        
        # Determine if overused (threshold: >15% or >3 times)
        is_overused = percentage > 15 or count > 3
        
        # Find positions where this opener was used
        positions = [pos for pos, op in opener_positions if op == opener]
        
        opener_analysis[opener] = {
            "count": count,
            "percentage": round(percentage, 1),
            "is_overused": is_overused,
            "severity": "high" if percentage > 25 else "medium" if percentage > 15 else "low",
            "positions": positions[:5],  # Show first 5 positions only
            "alternatives": OPENER_ALTERNATIVES.get(opener, [
                "Firstly", "In addition", "Moreover", "Furthermore", "Additionally"
            ])
        }
    
    # Generate recommendations
    recommendations = []
    overused_openers = {k: v for k, v in opener_analysis.items() if v["is_overused"]}
    
    if overused_openers:
        for opener, data in sorted(overused_openers.items(), key=lambda x: x[1]["count"], reverse=True):
            rec = {
                "opener": opener.title(),
                "usage": f"Used {data['count']} times ({data['percentage']}% of sentences)",
                "impact": "Reduces speech variety and can make delivery sound repetitive" if data["severity"] == "high" 
                         else "Moderately affects speech flow",
                "suggestion": f"Consider varied alternatives: {', '.join(data['alternatives'][:4])}",
                "severity": data["severity"]
            }
            recommendations.append(rec)
    
    # Determine overall status
    if not tracked_openers:
        status = "excellent"
        message = "✅ Great variety in sentence structure! No repetitive openers detected."
    elif not overused_openers:
        status = "good"
        message = "✓ Good sentence variety. Minor repetition present but not concerning."
    else:
        status = "needs_improvement"
        most_used = max(overused_openers.items(), key=lambda x: x[1]["count"])
        message = f"⚠️ '{most_used[0].title()}' is overused ({most_used[1]['count']} times, {most_used[1]['percentage']}%). Vary your sentence openers for better flow."
    
    logger.info(f"Sentence opener analysis complete: {len(tracked_openers)} unique openers found, {len(overused_openers)} overused")
    
    return {
        "status": status,
        "message": message,
        "openers_found": opener_analysis,
        "total_sentences": total_sentences,
        "tracked_openers_count": len(tracked_openers),
        "overused_count": len(overused_openers),
        "recommendations": recommendations,
        "variety_score": calculate_variety_score(opener_analysis, total_sentences)
    }


def calculate_variety_score(opener_analysis: Dict, total_sentences: int) -> float:
    """
    Calculate a variety score (0-100) based on sentence opener diversity
    
    Args:
        opener_analysis: Dictionary of opener analysis
        total_sentences: Total number of sentences
        
    Returns:
        Score from 0-100 (higher is better)
    """
    if not opener_analysis or total_sentences == 0:
        return 100.0  # Perfect if no repetition detected
    
    # Calculate penalty for overused openers
    penalty = 0
    for opener, data in opener_analysis.items():
        if data["is_overused"]:
            # Higher penalty for higher severity
            if data["severity"] == "high":
                penalty += 30
            elif data["severity"] == "medium":
                penalty += 20
            else:
                penalty += 10
    
    # Cap penalty at 70 (worst score is 30/100)
    penalty = min(penalty, 70)
    score = 100 - penalty
    
    return round(score, 1)
