"""
Report generator to integrate all metrics and generate feedback
"""
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.settings import (
    VOICE_CONFIDENCE_WEIGHT, FACIAL_CONFIDENCE_WEIGHT, VOCABULARY_WEIGHT,
    REPORTS_DIR
)
from ..services.scoring_engine import scoring_engine

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates comprehensive communication feedback reports"""
    
    def __init__(self):
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(exist_ok=True)
    
    def calculate_overall_score(self, voice_score: float, facial_score: float, vocabulary_score: float, is_audio_only: bool = False) -> float:
        """
        Calculate overall communication score
        
        Args:
            voice_score: Voice confidence score (0-100)
            facial_score: Facial confidence score (0-100)
            vocabulary_score: Vocabulary score (0-100)
            is_audio_only: Whether this is audio-only (no video)
            
        Returns:
            Overall communication score (0-100)
        """
        try:
            if is_audio_only or facial_score == 0:
                # For audio-only: redistribute weights (50% voice, 50% vocabulary)
                overall_score = (
                    voice_score * 0.5 +
                    vocabulary_score * 0.5
                )
            else:
                # For video: use standard weights
                overall_score = (
                    voice_score * VOICE_CONFIDENCE_WEIGHT +
                    facial_score * FACIAL_CONFIDENCE_WEIGHT +
                    vocabulary_score * VOCABULARY_WEIGHT
                )
            return round(overall_score, 1)
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 50.0
    
    def generate_strengths(self, audio_results: Dict, facial_results: Dict, text_results: Dict) -> List[str]:
        """
        Generate list of communication strengths based on actual analysis data
        
        Args:
            audio_results: Audio analysis results
            facial_results: Facial analysis results
            text_results: Text analysis results
            
        Returns:
            List of specific, data-driven strength statements
        """
        strengths = []
        
        try:
            # Get actual metrics
            speaking_metrics = audio_results.get('speaking_metrics', {})
            filler_analysis = audio_results.get('filler_analysis', {})
            pause_summary = audio_results.get('pause_summary', {})
            transcription = audio_results.get('transcription', {})
            
            # Voice strengths - be specific with numbers
            voice_score = audio_results.get('voice_confidence_score', 0)
            speaking_rate = transcription.get('speaking_rate_wpm', 0) or speaking_metrics.get('speaking_rate_wpm', 0)
            
            if voice_score >= 75:
                strengths.append(f"Strong voice delivery (score: {voice_score:.0f}/100)")
            
            if 140 <= speaking_rate <= 180:
                strengths.append(f"Optimal speaking pace ({speaking_rate:.0f} words/min)")
            elif 120 <= speaking_rate < 140:
                strengths.append(f"Clear and measured pace ({speaking_rate:.0f} words/min)")
            
            # Filler word analysis - specific counts
            total_fillers = filler_analysis.get('total_fillers', 0)
            filler_ratio = filler_analysis.get('filler_ratio', 0)
            total_words = transcription.get('word_count', 0) or speaking_metrics.get('total_words', 0)
            
            if total_fillers == 0 and total_words > 0:
                strengths.append("No filler words detected - excellent clarity")
            elif filler_ratio < 0.02 and total_words > 0:
                filler_pct = (filler_ratio * 100)
                strengths.append(f"Minimal filler usage ({filler_pct:.1f}% - {total_fillers} filler{'s' if total_fillers != 1 else ''} in {total_words} words)")
            
            # Pause analysis - specific data
            total_pauses = pause_summary.get('total_pauses', 0)
            avg_pause = pause_summary.get('avg_pause_duration', 0)
            if total_pauses > 0 and avg_pause < 1.5:
                strengths.append(f"Effective pause usage ({total_pauses} pauses, avg {avg_pause:.1f}s)")
            
            # Facial strengths - specific metrics
            if facial_results.get('analysis_successful', False):
                facial_score = facial_results.get('facial_confidence_score', 0)
                avg_eye_contact = facial_results.get('avg_eye_contact', 0)
                face_detection_rate = facial_results.get('face_detection_rate', 0)
                
                if facial_score >= 75:
                    strengths.append(f"Confident facial presence (score: {facial_score:.0f}/100)")
                
                if avg_eye_contact >= 0.7:
                    eye_pct = avg_eye_contact * 100
                    strengths.append(f"Strong eye contact ({eye_pct:.0f}% average)")
                elif avg_eye_contact >= 0.5:
                    eye_pct = avg_eye_contact * 100
                    strengths.append(f"Good eye contact maintained ({eye_pct:.0f}% average)")
                
                if face_detection_rate >= 0.9:
                    strengths.append(f"Consistent face visibility ({face_detection_rate*100:.0f}% of frames)")
                
                # Emotion analysis
                emotion_dist = facial_results.get('emotion_distribution', {})
                if emotion_dist:
                    dominant = max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else None
                    if dominant in ['happy', 'neutral'] and emotion_dist.get(dominant, 0) > 0.5:
                        strengths.append(f"Positive emotional expression ({dominant.capitalize()} dominant)")
            
            # Vocabulary strengths - specific metrics
            vocab_metrics = text_results.get('vocabulary_metrics', {})
            vocab_richness = vocab_metrics.get('vocabulary_richness', 0)
            unique_words = vocab_metrics.get('unique_words', 0)
            total_words_text = vocab_metrics.get('total_words', 0) or total_words
            
            if vocab_richness >= 0.5 and total_words_text > 0:
                strengths.append(f"Diverse vocabulary (richness: {vocab_richness:.2f}, {unique_words} unique words)")
            
            vocabulary_score = text_results.get('vocabulary_score', 0)
            if vocabulary_score >= 75:
                strengths.append(f"Strong vocabulary usage (score: {vocabulary_score:.0f}/100)")
            
            # If no specific strengths found, provide neutral feedback
            if len(strengths) == 0:
                strengths.append("Analysis completed - review specific metrics for detailed feedback")
            
        except Exception as e:
            logger.error(f"Error generating strengths: {e}")
            strengths = ["Analysis completed successfully"]
        
        return strengths
    
    def generate_improvements(self, audio_results: Dict, facial_results: Dict, text_results: Dict) -> List[str]:
        """
        Generate specific, data-driven improvement suggestions
        
        Args:
            audio_results: Audio analysis results
            facial_results: Facial analysis results
            text_results: Text analysis results
            
        Returns:
            List of specific improvement suggestions with actual metrics
        """
        improvements = []
        
        try:
            # Get actual metrics
            speaking_metrics = audio_results.get('speaking_metrics', {})
            filler_analysis = audio_results.get('filler_analysis', {})
            pause_summary = audio_results.get('pause_summary', {})
            transcription = audio_results.get('transcription', {})
            
            # Voice improvements - specific with numbers
            voice_score = audio_results.get('voice_confidence_score', 0)
            speaking_rate = transcription.get('speaking_rate_wpm', 0) or speaking_metrics.get('speaking_rate_wpm', 0)
            
            if voice_score < 70:
                improvements.append(f"Improve voice confidence (current: {voice_score:.0f}/100) - practice breathing exercises and vocal warm-ups")
            
            if speaking_rate < 120 and speaking_rate > 0:
                improvements.append(f"Increase speaking pace (current: {speaking_rate:.0f} wpm, target: 140-180 wpm) for better engagement")
            elif speaking_rate > 200:
                improvements.append(f"Slow down speaking pace (current: {speaking_rate:.0f} wpm, target: 140-180 wpm) for better clarity")
            
            # Filler word analysis - specific counts
            total_fillers = filler_analysis.get('total_fillers', 0)
            filler_ratio = filler_analysis.get('filler_ratio', 0)
            total_words = transcription.get('word_count', 0) or speaking_metrics.get('total_words', 0)
            filler_breakdown = filler_analysis.get('filler_breakdown', {})
            
            if filler_ratio > 0.04 and total_words > 0:
                filler_pct = (filler_ratio * 100)
                top_filler = max(filler_breakdown.items(), key=lambda x: x[1])[0] if filler_breakdown else None
                if top_filler:
                    improvements.append(f"Reduce filler words (current: {filler_pct:.1f}% - {total_fillers} instances). Most common: '{top_filler}' ({filler_breakdown[top_filler]}x)")
                else:
                    improvements.append(f"Reduce filler words (current: {filler_pct:.1f}% - {total_fillers} instances)")
            
            # Mumbling/Stammering detection
            mumbling_count = filler_analysis.get('mumbling_count', 0)
            stammering_count = filler_analysis.get('stammering_count', 0)
            
            if mumbling_count > 0:
                improvements.append(f"Improve articulation clarity - {mumbling_count} mumbling instance{'s' if mumbling_count != 1 else ''} detected")
            
            if stammering_count > 0:
                improvements.append(f"Practice smoother speech flow - {stammering_count} stammering instance{'s' if stammering_count != 1 else ''} detected")
            
            # Pause analysis - specific data
            total_pauses = pause_summary.get('total_pauses', 0)
            long_pauses = pause_summary.get('long_pauses', 0)
            longest_pause = pause_summary.get('longest_pause', 0)
            avg_pause = pause_summary.get('avg_pause_duration', 0)
            
            if long_pauses > 3:
                improvements.append(f"Reduce long pauses (detected {long_pauses} pauses >2s, longest: {longest_pause:.1f}s) - practice smoother transitions")
            elif total_pauses > 15 and avg_pause > 2.0:
                improvements.append(f"Optimize pause usage ({total_pauses} pauses, avg {avg_pause:.1f}s) - aim for shorter, strategic pauses")
            
            # Facial improvements - specific metrics
            if facial_results.get('analysis_successful', False):
                facial_score = facial_results.get('facial_confidence_score', 0)
                avg_eye_contact = facial_results.get('avg_eye_contact', 0)
                face_detection_rate = facial_results.get('face_detection_rate', 0)
                low_eye_contact_moments = facial_results.get('low_eye_contact_moments', [])
                tension_moments = facial_results.get('tension_moments', [])
                
                if facial_score < 70:
                    improvements.append(f"Improve facial confidence (current: {facial_score:.0f}/100) - practice confident expressions")
                
                if avg_eye_contact < 0.5:
                    eye_pct = avg_eye_contact * 100
                    improvements.append(f"Increase eye contact (current: {eye_pct:.0f}% average, target: >70%) - practice looking at camera/audience")
                
                if len(low_eye_contact_moments) > len(facial_results.get('eye_contact_timeline', [])) * 0.3:
                    improvements.append(f"Maintain consistent eye contact - {len(low_eye_contact_moments)} low-contact moments detected")
                
                if face_detection_rate < 0.7:
                    improvements.append(f"Improve face visibility (detected in {face_detection_rate*100:.0f}% of frames) - ensure good lighting and camera angle")
                
                if len(tension_moments) > 0:
                    tension_pct = (len(tension_moments) / len(facial_results.get('frame_results', [1]))) * 100 if facial_results.get('frame_results') else 0
                    if tension_pct > 20:
                        improvements.append(f"Reduce tension/stress moments ({len(tension_moments)} detected, {tension_pct:.0f}% of video) - practice relaxation techniques")
                
                # Emotion analysis
                emotion_dist = facial_results.get('emotion_distribution', {})
                negative_emotions = ['angry', 'sad', 'fear', 'disgust']
                negative_count = sum(emotion_dist.get(emotion, 0) for emotion in negative_emotions)
                total_emotions = sum(emotion_dist.values()) if emotion_dist else 1
                if negative_count / total_emotions > 0.3:
                    improvements.append(f"Project more positive emotions - {negative_count}/{total_emotions} frames showed negative emotions")
            
            # Vocabulary improvements - specific metrics
            vocab_metrics = text_results.get('vocabulary_metrics', {})
            vocab_richness = vocab_metrics.get('vocabulary_richness', 0)
            unique_words = vocab_metrics.get('unique_words', 0)
            total_words_text = vocab_metrics.get('total_words', 0) or total_words
            
            vocabulary_score = text_results.get('vocabulary_score', 0)
            if vocabulary_score < 70:
                improvements.append(f"Enhance vocabulary usage (current: {vocabulary_score:.0f}/100) - use more precise and varied words")
            
            if vocab_richness < 0.4 and total_words_text > 0:
                improvements.append(f"Increase vocabulary diversity (current richness: {vocabulary_richness:.2f}, {unique_words}/{total_words_text} unique words)")
            
            # Word analysis improvements
            word_analysis = audio_results.get('word_analysis', {})
            weak_words_data = word_analysis.get('weak_words', {})
            weak_word_count = weak_words_data.get('weak_word_count', 0)
            if weak_word_count > 5:
                improvements.append(f"Replace weak words ({weak_word_count} instances) with stronger, more specific language")
            
            # If no specific improvements found
            if len(improvements) == 0:
                improvements.append("Continue practicing to maintain and refine your communication skills")
            
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
            improvements = ["Focus on consistent practice based on your specific metrics"]
        
        return improvements
    
    def format_emotion_distribution(self, emotion_distribution: Dict) -> Dict:
        """
        Format emotion distribution for display
        
        Args:
            emotion_distribution: Raw emotion distribution
            
        Returns:
            Formatted emotion distribution
        """
        try:
            # Convert to percentages and round
            total = sum(emotion_distribution.values())
            if total == 0:
                return {}
            
            formatted = {}
            for emotion, count in emotion_distribution.items():
                percentage = (count / total) * 100
                formatted[emotion] = round(percentage, 1)
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting emotion distribution: {e}")
            return {}
    
    def generate_report(self, session_id: str, audio_results: Dict, facial_results: Dict, text_results: Dict, file_type: str = "audio", word_analysis: Dict = None, transcript_improvement: Dict = None, vocabulary_enhancements: Dict = None) -> Dict:
        """
        Generate complete communication feedback report
        
        Args:
            session_id: Session identifier
            audio_results: Audio analysis results
            facial_results: Facial analysis results
            text_results: Text analysis results
            file_type: Type of file analyzed ("audio" or "video")
            
        Returns:
            Complete report dictionary
        """
        try:
            logger.info(f"Generating report for session: {session_id} (type: {file_type})")
            
            # Extract key metrics from enhanced audio analysis
            logger.info(f"Extracting scores - audio keys: {list(audio_results.keys())}")
            logger.info(f"Extracting scores - facial keys: {list(facial_results.keys())}")
            logger.info(f"Extracting scores - text keys: {list(text_results.keys())}")
            
            # CRITICAL FIX: Robust extraction of voice_confidence_score with multiple fallbacks
            voice_score = None
            
            # Try 1: Direct voice_confidence_score from audio_results
            if 'voice_confidence_score' in audio_results:
                voice_score = audio_results.get('voice_confidence_score')
                logger.info(f"Found voice_confidence_score directly: {voice_score}")
            
            # Try 2: From voice_confidence_breakdown (if available)
            if voice_score is None or voice_score == 0:
                breakdown = audio_results.get('voice_confidence_breakdown', {})
                if breakdown and 'final_score' in breakdown:
                    voice_score = breakdown.get('final_score')
                    logger.info(f"Found voice_confidence_score from breakdown: {voice_score}")
            
            # Try 3: From speaking_metrics
            if voice_score is None or voice_score == 0:
                speaking_metrics = audio_results.get('speaking_metrics', {})
                voice_score = speaking_metrics.get('voice_score') or speaking_metrics.get('voice_confidence_score')
                if voice_score:
                    logger.info(f"Found voice_confidence_score from speaking_metrics: {voice_score}")
            
            # Try 4: Calculate a basic score if we have data
            if voice_score is None or voice_score == 0:
                # Calculate basic score from available metrics
                speaking_rate = audio_results.get('speaking_metrics', {}).get('speaking_rate_wpm', 0)
                filler_ratio = audio_results.get('filler_analysis', {}).get('filler_ratio', 0) or \
                              audio_results.get('filler_word_ratio', 0)
                
                # Basic heuristic: start with 50, adjust based on available data
                voice_score = 50.0
                if speaking_rate > 0:
                    if 120 <= speaking_rate <= 180:
                        voice_score += 15
                    elif 100 <= speaking_rate < 120 or 180 < speaking_rate <= 200:
                        voice_score += 5
                    elif speaking_rate < 90 or speaking_rate > 210:
                        voice_score -= 10
                
                if filler_ratio > 0:
                    if filler_ratio < 0.05:
                        voice_score += 10
                    elif filler_ratio < 0.10:
                        voice_score += 5
                    elif filler_ratio > 0.15:
                        voice_score -= 15
                
                voice_score = max(0, min(100, voice_score))
                logger.warning(f"Calculated fallback voice_confidence_score: {voice_score} (from speaking_rate={speaking_rate}, filler_ratio={filler_ratio})")
            
            # Final fallback: ensure we never return 0 unless truly no data
            if voice_score is None or voice_score == 0:
                # Last resort: use a default score based on whether we have any audio data
                has_transcript = bool(audio_results.get('transcript') or audio_results.get('words_with_timing', []))
                voice_score = 35.0 if has_transcript else 0.0
                logger.warning(f"Using final fallback voice_confidence_score: {voice_score} (has_transcript={has_transcript})")
            
            facial_score = facial_results.get('facial_confidence_score', 0)
            vocabulary_score = text_results.get('vocabulary_score', 0)
            
            logger.info(f"Final extracted scores - voice: {voice_score}, facial: {facial_score}, vocab: {vocabulary_score}")
            
            # Determine if audio-only
            is_audio_only = (file_type == "audio") or (facial_score == 0)
            
            # Calculate overall score using the composite scoring engine for better calibration
            scorecard = scoring_engine.evaluate(
                audio_results=audio_results,
                facial_results=facial_results,
                text_results=text_results,
                file_type=("audio" if is_audio_only else "video")
            )
            overall_score = round(scorecard.get("composite", 0.0), 1)
            
            # Extract detailed metrics
            audio_features = audio_results.get('audio_features', {})
            clarity_metrics = audio_results.get('clarity_metrics', {})
            transcription = audio_results.get('transcription', {})
            
            # Generate feedback
            strengths = self.generate_strengths(audio_results, facial_results, text_results)
            improvements = self.generate_improvements(audio_results, facial_results, text_results)
            
            # Add specific feedback based on enhanced analysis
            filler_analysis = audio_results.get('filler_analysis', {})
            pause_summary = audio_results.get('pause_summary', {})
            
            # Add pause-specific feedback
            if pause_summary.get('longest_pause', 0) >= 5:
                improvements.insert(0, f"⚠️ Detected {pause_summary.get('longest_pause', 0):.1f}s gap - practice reducing long pauses")
            
            # Add filler-specific feedback
            if filler_analysis.get('total_fillers', 0) > 5:
                filler_breakdown = filler_analysis.get('filler_breakdown', {})
                top_filler = max(filler_breakdown.items(), key=lambda x: x[1])[0] if filler_breakdown else "um"
                improvements.insert(0, f"⚠️ Used '{top_filler}' {filler_breakdown.get(top_filler, 0)} times - practice eliminating filler words")
            
            # Format emotion distribution
            emotion_distribution = self.format_emotion_distribution(
                facial_results.get('emotion_distribution', {})
            )
            
            # DEBUG: Log all scores
            logger.info(f"SCORES - Voice: {voice_score}, Facial: {facial_score}, Vocab: {vocabulary_score}")
            logger.info(f"Overall: {overall_score}, Audio-only: {is_audio_only}")
            
            # Create report
            report = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "file_type": file_type,
                "is_audio_only": is_audio_only,
                "duration": audio_results.get('speaking_metrics', {}).get('total_duration', 0),
                "overall_score": overall_score,
                "scorecard": scorecard,
                "voice_confidence": round(float(voice_score or 0), 1),
                "facial_confidence": round(facial_score, 1) if not is_audio_only and facial_score else 0,
                "vocabulary_score": round(vocabulary_score, 1) if vocabulary_score else 0,
                
                # Detailed metrics
                "filler_word_count": clarity_metrics.get('filler_word_count', 0),
                "filler_word_ratio": round(clarity_metrics.get('filler_word_ratio', 0), 3),
                "pause_count": audio_features.get('total_pauses', 0),
                "long_pauses": audio_features.get('long_pauses', 0),
                "speaking_rate_wpm": round(transcription.get('speaking_rate_wpm', 0), 1),
                "voice_breaks_ratio": round(audio_features.get('voice_breaks_ratio', 0), 3),
                
                # Facial metrics
                "dominant_emotion": facial_results.get('dominant_emotion', 'neutral'),
                "emotion_distribution": emotion_distribution,
                "avg_eye_contact": round(facial_results.get('avg_eye_contact', 0), 2),
                "face_detection_rate": round(facial_results.get('face_detection_rate', 0), 2),
                
                # NEW: Precise timestamp tracking for facial analysis
                "emotion_timeline": facial_results.get('emotion_timeline', []),
                "eye_contact_timeline": facial_results.get('eye_contact_timeline', []),
                "tension_moments": facial_results.get('tension_moments', []),
                "low_eye_contact_moments": facial_results.get('low_eye_contact_moments', []),
                "high_eye_contact_moments": facial_results.get('high_eye_contact_moments', []),
                "tension_count": facial_results.get('tension_count', 0),
                "tension_percentage": round(facial_results.get('tension_percentage', 0), 2),
                
                # Text metrics
                "vocabulary_richness": round(text_results.get('vocabulary_metrics', {}).get('vocabulary_richness', 0), 3),
                "unique_words": text_results.get('vocabulary_metrics', {}).get('unique_words', 0),
                "total_words": text_results.get('vocabulary_metrics', {}).get('total_words', 0),
                "avg_sentence_length": round(text_results.get('structure_metrics', {}).get('avg_sentence_length', 0), 1),
                
                # Content
                "transcript": audio_results.get('transcript', transcription.get('transcript', '')),
                # VERBATIM: Store raw word-level timing for transcript reconstruction
                "words_with_timing": audio_results.get('words_with_timing', []),
                "enhanced_transcript": audio_results.get('enhanced_transcript', 
                                                        audio_results.get('clarity_metrics', {}).get('enhanced_transcript', '')),
                "pause_analysis": audio_results.get('pause_summary', 
                                                   audio_results.get('clarity_metrics', {}).get('pause_analysis', {})),
                "pauses_detailed": audio_results.get('pauses', []),
                "pitch_data": audio_results.get('pitch_data', {}),
                "filler_analysis": audio_results.get('filler_analysis', {}),
                "speaking_metrics": audio_results.get('speaking_metrics', {}),
                
                # NEW: Precise stammering/mumbling tracking with timestamps
                "mumbling_instances": audio_results.get('filler_analysis', {}).get('mumbling_instances', []),
                "stammering_instances": audio_results.get('filler_analysis', {}).get('stammering_instances', []),
                "mumbling_clusters": audio_results.get('filler_analysis', {}).get('mumbling_clusters', []),
                "stammering_clusters": audio_results.get('filler_analysis', {}).get('stammering_clusters', []),
                "mumbling_count": audio_results.get('filler_analysis', {}).get('mumbling_count', 0),
                "stammering_count": audio_results.get('filler_analysis', {}).get('stammering_count', 0),
                
                # Word analysis (weak words, fillers, vocabulary)
                "word_analysis": word_analysis if word_analysis else {},
                
                # Transcript and vocabulary improvements
                "transcript_improvement": transcript_improvement if transcript_improvement else {},
                "vocabulary_enhancements": vocabulary_enhancements if vocabulary_enhancements else {},
                "summary": text_results.get('summary', ''),
                "strengths": strengths,
                "improvements": improvements,
                
                # Strict evaluation data (if available)
                "strict_evaluation": text_results.get('strict_evaluation'),
                
                # Analysis status
                "analysis_successful": all([
                    audio_results.get('analysis_successful', False),
                    facial_results.get('analysis_successful', False),
                    text_results.get('analysis_successful', False)
                ])
            }

            # Ensure audio_analytics always has pause_cadence structure
            advanced_audio = audio_results.get("advanced_audio_metrics", {})
            if not advanced_audio.get("pause_cadence"):
                # Ensure pause_cadence exists even if empty
                advanced_audio["pause_cadence"] = {
                    "counts": {"short": 0, "medium": 0, "long": 0},
                    "durations": {"short": 0.0, "medium": 0.0, "long": 0.0},
                    "average_duration": 0.0,
                    "total_pause_time": 0.0
                }
            report["audio_analytics"] = advanced_audio
            report["visual_analytics"] = {
                "tension_summary": facial_results.get("tension_summary", {}),
                "emotion_timeline_smoothed": facial_results.get("emotion_timeline_smoothed", []),
                "eye_contact_stability": (facial_results.get("tension_summary") or {}).get("eye_contact_stability"),
                "avg_eye_contact_pct": (facial_results.get("tension_summary") or {}).get("avg_eye_contact_pct"),
            }
            report["text_analytics"] = text_results.get("advanced_text_metrics", {})
            
            # Save report to file
            report_path = self.reports_dir / f"{session_id}.json"
            clean_report = self._sanitize_for_json(report)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(clean_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report saved to: {report_path}")
            return clean_report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            fallback = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "overall_score": 0,
                "error": str(e),
                "analysis_successful": False
            }
            return self._sanitize_for_json(fallback)
    
    def build_session_summary(
        self,
        session_id: str,
        report: Dict,
        audio_results: Dict,
        facial_results: Dict,
        text_results: Dict,
        metadata: Dict = None
    ) -> Dict:
        """
        Construct a condensed session summary for persistence/history views.
        """
        metadata = metadata or {}

        score_breakdown = {
            "voice_confidence": report.get("voice_confidence"),
            "facial_confidence": report.get("facial_confidence"),
            "vocabulary_score": report.get("vocabulary_score"),
        }

        metrics = {
            "duration": report.get("duration") or 0,
            "filler_word_count": report.get("filler_word_count"),
            "filler_word_ratio": report.get("filler_word_ratio"),
            "pause_count": report.get("pause_count"),
            "long_pauses": report.get("long_pauses"),
            "speaking_rate_wpm": report.get("speaking_rate_wpm"),
            "avg_pause_duration": audio_results.get("pause_summary", {}).get("avg_pause_duration"),
            "dominant_emotion": report.get("dominant_emotion"),
            "avg_eye_contact": report.get("avg_eye_contact"),
            "vocabulary_richness": report.get("vocabulary_richness"),
        }

        advanced_audio = audio_results.get("advanced_audio_metrics", {})
        if advanced_audio:
            metrics["opening_confidence"] = advanced_audio.get("opening_confidence", {}).get("opening_confidence")
            metrics["pause_profile"] = advanced_audio.get("pause_cadence", {}).get("counts")
            metrics["top_fillers"] = advanced_audio.get("filler_trend", {}).get("top_labels")

        tension_summary = facial_results.get("tension_summary") or {}
        if tension_summary:
            metrics["tension_percentage"] = tension_summary.get("tension_percentage")
            metrics["eye_contact_stability"] = tension_summary.get("eye_contact_stability")

        advanced_text = text_results.get("advanced_text_metrics", {})
        if advanced_text:
            metrics["topic_coherence_score"] = advanced_text.get("topic_coherence_score")
            metrics["keyword_coverage"] = advanced_text.get("keyword_coverage", {}).get("keyword_density")

        thumbnail_url = metadata.get("thumbnail_url") or report.get("thumbnail_url")
        highlights = {
            "strengths": (report.get("strengths") or [])[:3],
            "growth_areas": (report.get("improvements") or [])[:3],
        }
        if thumbnail_url:
            highlights["thumbnail_url"] = thumbnail_url

        title = metadata.get("title") or metadata.get("file_name") or f"Session {session_id[:8]}"

        summary = {
            "session_id": session_id,
            "user_id": metadata.get("user_id"),
            "title": title,
            "file_name": metadata.get("file_name", title),
            "file_type": metadata.get("file_type") or report.get("file_type"),
            "overall_score": report.get("overall_score"),
            "score_breakdown": score_breakdown,
            "metrics": metrics,
            "highlights": highlights,
            "created_at": report.get("timestamp"),
            "pdf_path": metadata.get("pdf_path"),
        }

        return summary
    
    def load_report(self, session_id: str) -> Dict:
        """
        Load existing report from file
        
        Args:
            session_id: Session identifier
            
        Returns:
            Report dictionary or empty dict if not found
        """
        try:
            report_path = self.reports_dir / f"{session_id}.json"
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except json.JSONDecodeError as decode_error:
            logger.error(f"Corrupted report file for session {session_id}: {decode_error}")
            try:
                report_path = self.reports_dir / f"{session_id}.json"
                if report_path.exists():
                    report_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Failed to remove corrupted report {report_path}: {cleanup_error}")
            return {}
        except Exception as e:
            logger.error(f"Error loading report: {e}")
            return {}

    def _sanitize_for_json(self, value: Any) -> Any:
        """
        Recursively convert values to JSON-serializable primitives.
        Handles numpy scalars/arrays, sets, and pathlib paths.
        """
        if isinstance(value, dict):
            return {k: self._sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, set):
            return [self._sanitize_for_json(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

# Global report generator instance
report_generator = ReportGenerator()
