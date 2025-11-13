"""
Detailed feedback generator for comprehensive presentation analysis
Generates Strength, Growth Area, Follow-up Questions, Tone, Visual Presence, Conciseness, Summary, and Pronunciation feedback
"""
import logging
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple
from models.openai_enhancer import openai_enhancer
from models.report_generator import report_generator
from models.pronunciation_analyzer import pronunciation_analyzer
from models.transcript_enhancer import transcript_enhancer
from models.vocabulary_enhancer import vocabulary_enhancer

logger = logging.getLogger(__name__)

class DetailedFeedbackGenerator:
    """Generates detailed coaching feedback for presentation analysis"""
    
    def __init__(self):
        self.openai = openai_enhancer

    # ------------------------------------------------------------------
    # Helpers to keep the coaching output concise and learner-facing
    # ------------------------------------------------------------------

    def _normalize_strength(self, raw_strength) -> Optional[Dict]:
        if isinstance(raw_strength, dict):
            message = raw_strength.get("message") or raw_strength.get("text")
        elif isinstance(raw_strength, str):
            message = raw_strength
        else:
            message = None
        if message:
            return {"message": message.strip()}
        return None

    def _normalize_growth_areas(self, raw_growth) -> List[str]:
        if not raw_growth:
            return []
        if isinstance(raw_growth, str):
            return [raw_growth.strip()]
        if isinstance(raw_growth, list):
            return [str(item).strip() for item in raw_growth if str(item).strip()]
        return []

    def _normalize_questions(self, raw_questions) -> List[Dict]:
        normalized = []
        if isinstance(raw_questions, list):
            for item in raw_questions[:5]:
                if isinstance(item, dict) and item.get("question"):
                    normalized.append({
                        "question": str(item["question"]).strip(),
                        "timestamp": str(item.get("timestamp", "")).strip() or None
                    })
                elif isinstance(item, str):
                    normalized.append({"question": item.strip(), "timestamp": None})
        return normalized

    def _build_metrics_summary(self, report_data: Dict, pronunciation_score: Optional[float] = None) -> Dict:
        filler_count = report_data.get("filler_word_count") or report_data.get("filler_analysis", {}).get("total_fillers") or 0
        total_words = (
            report_data.get("total_words")
            or report_data.get("vocabulary_metrics", {}).get("total_words")
            or report_data.get("speaking_metrics", {}).get("total_words")
            or 0
        )
        weak_words_data = report_data.get("word_analysis", {}).get("weak_words", {})
        weak_word_count = weak_words_data.get("weak_word_count") or 0
        speaking_rate = report_data.get("speaking_rate_wpm") or report_data.get("speaking_metrics", {}).get("speaking_rate_wpm")

        filler_percentage = round((filler_count / total_words) * 100, 1) if total_words else 0.0
        weak_word_percentage = weak_words_data.get("weak_word_percentage")
        if weak_word_percentage is None and total_words:
            weak_word_percentage = round((weak_word_count / total_words) * 100, 1)

        conciseness_score = report_data.get("speaking_metrics", {}).get("conciseness_score")
        conciseness_note = None
        if conciseness_score is None and speaking_rate:
            conciseness_score = max(0, min(100, 100 - abs(speaking_rate - 145)))
        if speaking_rate:
            if speaking_rate > 170:
                conciseness_note = "Speaking pace was fast; consider pausing for emphasis."
            elif speaking_rate < 120:
                conciseness_note = "Speaking pace was slow; add energy and shorten sentences."
            else:
                conciseness_note = "Speaking pace sat in the conversational range."

        summary = {
            "filler": {
                "count": int(filler_count),
                "percentage": float(filler_percentage),
            },
            "weak_words": {
                "count": int(weak_word_count),
                "percentage": float(weak_word_percentage) if weak_word_percentage is not None else None,
            },
            "conciseness": {
                "score": float(conciseness_score) if conciseness_score is not None else None,
                "notes": conciseness_note,
            },
        }
        if pronunciation_score is not None:
            summary["pronunciation"] = {
                "score": float(pronunciation_score),
                "note": "Pronunciation clarity improved with fewer flagged words." if pronunciation_score >= 70 else None
            }
        return summary

    def _summarize_pronunciation(self, pronunciation_analysis: Dict, raw_section: Optional[Dict]) -> Dict:
        issues = pronunciation_analysis.get("issues", []) if pronunciation_analysis else []
        summary_text = None
        if raw_section:
            summary_text = raw_section.get("summary") or raw_section.get("message")
        if not summary_text:
            if issues:
                summary_text = f"Detected {len(issues)} pronunciation targets worth practising."
            else:
                summary_text = "No major pronunciation issues detected."

        normalized_issues = []
        for issue in issues[:5]:
            if not isinstance(issue, dict):
                continue
            word = issue.get("word")
            correct = issue.get("correct")
            tip = issue.get("context")
            if word:
                normalized_issues.append({
                    "word": word,
                    "correct": correct,
                    "tip": tip
                })

        return {
            "summary": summary_text.strip(),
            "issues": normalized_issues
        }

    def _coerce_feedback_schema(
        self,
        raw_feedback: Dict,
        report_data: Dict,
        pronunciation_analysis: Dict,
        pronunciation_score: Optional[float] = None
    ) -> Dict:
        """Ensure we only surface learner-facing coaching data."""
        feedback = {}

        strength = self._normalize_strength(raw_feedback.get("strength"))
        if strength:
            feedback["strength"] = strength

        growth_areas = self._normalize_growth_areas(raw_feedback.get("growth_areas"))
        if growth_areas:
            feedback["growth_areas"] = growth_areas

        questions = self._normalize_questions(raw_feedback.get("follow_up_questions"))
        if questions:
            feedback["follow_up_questions"] = questions

        metrics = raw_feedback.get("metrics") if isinstance(raw_feedback.get("metrics"), dict) else {}
        computed_metrics = self._build_metrics_summary(report_data, pronunciation_score)
        metrics = {**computed_metrics, **metrics} if metrics else computed_metrics
        feedback["metrics"] = metrics

        pronunciation_section = raw_feedback.get("pronunciation") if isinstance(raw_feedback.get("pronunciation"), dict) else {}
        feedback["pronunciation"] = self._summarize_pronunciation(pronunciation_analysis, pronunciation_section)

        summary = raw_feedback.get("summary")
        if isinstance(summary, list):
            summary = " ".join([str(item).strip() for item in summary if str(item).strip()])
        if isinstance(summary, dict):
            summary = summary.get("text") or summary.get("message")
        if not summary:
            # fall back to concise auto summary
            summary_lines = []
            if strength and strength.get("message"):
                summary_lines.append(f"Strength: {strength['message']}")
            if growth_areas:
                summary_lines.append(f"Focus next: {growth_areas[0]}")
            filler_pct = metrics.get("filler", {}).get("percentage")
            if filler_pct is not None:
                summary_lines.append(f"Filler usage: {filler_pct:.1f}% of words.")
            summary = " ".join(summary_lines) if summary_lines else "Keep refining your delivery with focused practice."
        feedback["summary"] = summary.strip()

        highlights = raw_feedback.get("highlights")
        if isinstance(highlights, list):
            cleaned = []
            for item in highlights:
                if isinstance(item, dict) and "note" in item:
                    cleaned.append({
                        "t": float(item.get("t", 0.0)),
                        "note": str(item["note"]).strip()
                    })
            if cleaned:
                feedback["highlights"] = cleaned

        # Strip any residual verbose fields
        return feedback
    
    def generate_comprehensive_feedback(self, session_id: str, report_data: Dict) -> Dict:
        """
        Generate comprehensive feedback including all coaching sections
        
        Args:
            session_id: Session identifier
            report_data: Complete report data from analysis
            
        Returns:
            Dictionary with all feedback sections
        """
        try:
            logger.info(f"Generating comprehensive feedback for session: {session_id}")
            
            transcript = report_data.get('transcript', '')
            enhanced_transcript = report_data.get('enhanced_transcript', transcript)
            
            # If OpenAI is available, use it for enhanced feedback
            if self.openai.enabled:
                return self._generate_ai_feedback(report_data, transcript, enhanced_transcript)
            else:
                # Fallback to rule-based feedback
                return self._generate_rule_based_feedback(report_data, transcript)
                
        except Exception as e:
            logger.error(f"Error generating comprehensive feedback: {e}")
            return self._generate_fallback_feedback(report_data)
    
    def _generate_ai_feedback(self, report_data: Dict, transcript: str, enhanced_transcript: str) -> Dict:
        """Generate feedback using OpenAI API"""
        try:
            # Run pronunciation analysis
            pronunciation_analysis = pronunciation_analyzer.analyze_transcript(transcript)
            pronunciation_issues = pronunciation_analysis.get('issues', []) if isinstance(pronunciation_analysis, dict) else []
            pronunciation_score = max(
                35,
                min(
                    95,
                    100 - max(0, len(pronunciation_issues) * 8)
                )
            )
            
            # Run transcript enhancement (if not already done)
            if 'transcript_improvement' not in report_data:
                transcript_improvement = transcript_enhancer.enhance_transcript(transcript)
            else:
                transcript_improvement = report_data.get('transcript_improvement', {})
            
            # Run vocabulary enhancement (if not already done)
            if 'vocabulary_enhancements' not in report_data:
                vocabulary_enhancements = vocabulary_enhancer.enhance_vocabulary(transcript)
            else:
                vocabulary_enhancements = report_data.get('vocabulary_enhancements', {})
            
            # Prepare context for OpenAI
            context = self._prepare_context(report_data, transcript)
            
            prompt = f"""You are an expert presentation coach analyzing a recorded presentation. Based on the following analysis data, provide comprehensive feedback in the specified JSON format.

ANALYSIS DATA:
{context}

TRANSCRIPT:
{enhanced_transcript[:3000]}

Please provide feedback in this exact JSON format:
{{
    "strength": {{
        "message": "A clear, specific strength statement",
        "icon": "lightbulb"
    }},
    "growth_areas": [
        "First specific growth area with actionable advice",
        "Second specific growth area with actionable advice",
        "Third specific growth area (if applicable)"
    ],
    "follow_up_questions": [
        {{
            "question": "First follow-up question",
            "timestamp": "0:22"
        }},
        {{
            "question": "Second follow-up question",
            "timestamp": "0:27"
        }},
        {{
            "question": "Third follow-up question",
            "timestamp": "0:30"
        }}
    ],
    "tone": {{
        "assessment": "Under Pressure" | "Confident" | "Calm" | "Neutral",
        "description": "Brief explanation of tone assessment"
    }},
    "visual_presence": {{
        "primary": "Static" | "Reserved" | "Tense" | "Focused",
        "secondary": ["Option1", "Option2"],
        "description": "Brief explanation of visual presence"
    }},
    "conciseness": {{
        "score": 0-100,
        "concise_version": "A more concise version of key points (first 200 words)",
        "suggestions": ["Suggestion 1", "Suggestion 2"]
    }},
    "summary": {{
        "points": [
            "First key point about the presentation",
            "Second key point about the presentation",
            "Third key point (if applicable)"
        ]
    }},
    "pronunciation": {{
        "issues": [
            {{
                "word": "Python",
                "incorrect": "pyTHON",
                "correct": "PYthon",
                "phonetic": "/ˈpaɪθɑːn/",
                "context": "When you said 'navigation system using Python'"
            }},
            {{
                "word": "database",
                "incorrect": "daTAbase",
                "correct": "DA-ta-base",
                "phonetic": "/ˈdeɪtəbeɪs/",
                "context": "When you said 'and for database, we have used MySQL'"
            }}
        ],
        "general_advice": "General pronunciation advice focusing on syllable stress"
    }}
}}

Focus on:
- Strength: Highlight what was done well (technical knowledge, clarity, etc.)
- Growth Areas: Be specific about unclear descriptions, branding issues, etc.
- Follow-up Questions: Prepare questions that interviewers might ask (3-5 questions)
- Tone: Assess if speaker sounds under pressure, confident, or calm
- Visual Presence: Assess if static, reserved, tense, or focused
- Conciseness: Provide a more concise version if needed
- Summary: Key points about the presentation
- Pronunciation: Focus on syllable stress issues in American English

Respond ONLY with valid JSON, no additional text."""

            response = self.openai.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert presentation coach. Always respond with valid JSON only, no markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            feedback_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if feedback_text.startswith("```"):
                feedback_text = feedback_text.split("```")[1]
                if feedback_text.startswith("json"):
                    feedback_text = feedback_text[4:]
            
            raw_feedback = json.loads(feedback_text)
            
            coach_feedback = self._coerce_feedback_schema(
                raw_feedback,
                report_data,
                pronunciation_analysis,
                pronunciation_score
            )
            # Attach supporting enhancements for UI components that still rely on them
            coach_feedback['transcript_improvement'] = transcript_improvement
            coach_feedback['vocabulary_enhancements'] = vocabulary_enhancements
            
            logger.info("AI feedback generated successfully with pronunciation, transcript, and vocabulary analysis")
            return coach_feedback
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in AI feedback: {e}")
            logger.error(f"Response was: {feedback_text[:500]}")
            feedback = self._generate_rule_based_feedback(report_data, transcript)
            return feedback
        except Exception as e:
            logger.error(f"Error generating AI feedback: {e}")
            return self._generate_rule_based_feedback(report_data, transcript)
    
    def _generate_rule_based_feedback(self, report_data: Dict, transcript: str) -> Dict:
        """Generate feedback using rule-based approach as fallback"""
        try:
            strengths = report_data.get('strengths', [])
            improvements = report_data.get('improvements', [])
            
            filler_feedback = self._build_filler_feedback(report_data)
            pace_feedback = self._build_pace_feedback(report_data)
            voice_feedback = self._build_voice_feedback(report_data)

            acoustic_strength = filler_feedback.get('strength')
            acoustic_growth = filler_feedback.get('growth', [])

            pace_growth = pace_feedback.get('growth')
            pace_strength = pace_feedback.get('strength')

            voice_growth = voice_feedback.get('growth')
            voice_strength = voice_feedback.get('strength')

            visual_feedback = self._build_visual_feedback(report_data)

            # Strength message: prefer refined strengths from metrics, fallback to provided list
            computed_strengths: List[str] = []
            if acoustic_strength:
                computed_strengths.append(acoustic_strength)
            if pace_strength:
                computed_strengths.append(pace_strength)
            if voice_strength:
                computed_strengths.append(voice_strength)
            if visual_feedback.get('strength'):
                computed_strengths.append(visual_feedback['strength'])

            if not computed_strengths and strengths:
                computed_strengths.append(strengths[0])
            elif not computed_strengths:
                computed_strengths.append("You maintained solid delivery fundamentals throughout the presentation.")

            strength_message = computed_strengths[0]

            # Growth areas
            growth_areas: List[str] = []
            growth_areas.extend(acoustic_growth)
            if pace_growth:
                growth_areas.append(pace_growth)
            if voice_growth:
                growth_areas.append(voice_growth)
            growth_areas.extend(visual_feedback.get('growth', []))

            if not growth_areas:
                growth_areas = (
                    improvements[:3]
                    if improvements
                    else [
                        "Clarify your technical descriptions for better audience understanding.",
                        "Ensure project descriptions align with personal branding."
                    ]
                )
            else:
                # Deduplicate growth areas while preserving order
                seen = set()
                deduped_growth = []
                for item in growth_areas:
                    if item and item not in seen:
                        seen.add(item)
                        deduped_growth.append(item)
                growth_areas = deduped_growth[:4]
            
            # Follow-up Questions
            follow_up_questions = [
                {"question": "Can you elaborate on how the SOS alerts and alternate road navigation system work in your AI-powered app?", "timestamp": "0:22"},
                {"question": "What challenges did you face during development?", "timestamp": "0:27"},
                {"question": "How did you test the safety features of your application?", "timestamp": "0:30"}
            ]
            
            # Tone assessment with descriptive context
            tone_details = self._build_tone_feedback(report_data)
            
            visual_presence_primary = visual_feedback.get('primary', 'Static')
            visual_presence_description = visual_feedback.get('description', f"Visual presence assessment: {visual_presence_primary.lower()}.")
            
            # Conciseness
            total_words = report_data.get('total_words', 0)
            conciseness_score = 100 if total_words < 200 else max(50, 100 - (total_words - 200) // 10)
            
            # Summary
            summary_points = self._build_summary(report_data, filler_feedback, pace_feedback, visual_feedback)
            
            # Run pronunciation analysis
            pronunciation_analysis = pronunciation_analyzer.analyze_transcript(transcript)
            pronunciation_issues = pronunciation_analysis.get('issues', [])
            pronunciation_score = max(
                35,
                min(
                    95,
                    100 - max(0, len(pronunciation_issues) * 8)
                )
            )
            
            # Run transcript enhancement
            transcript_improvement = transcript_enhancer.enhance_transcript(transcript)
            
            # Run vocabulary enhancement
            vocabulary_enhancements = vocabulary_enhancer.enhance_vocabulary(transcript)
            
            raw_feedback = {
                "strength": {"message": strength_message},
                "growth_areas": growth_areas,
                "follow_up_questions": follow_up_questions,
                "pronunciation": {
                    "summary": pronunciation_analysis.get('general_advice'),
                    "issues": pronunciation_issues
                },
                "metrics": self._build_metrics_summary(report_data, pronunciation_score),
                "summary": " ".join(summary_points) if summary_points else None,
                "highlights": [
                    {"t": ts, "note": "Consider addressing this filler moment"}
                    for ts in filler_feedback.get("timestamps", []) if isinstance(ts, (int, float))
                ]
            }

            coach_feedback = self._coerce_feedback_schema(
                raw_feedback,
                report_data,
                pronunciation_analysis,
                pronunciation_score
            )
            coach_feedback["transcript_improvement"] = transcript_improvement
            coach_feedback["vocabulary_enhancements"] = vocabulary_enhancements
            return coach_feedback
            
        except Exception as e:
            logger.error(f"Error generating rule-based feedback: {e}")
            return self._generate_fallback_feedback(report_data)
    
    def _extract_filler_stats(self, report_data: Dict) -> Dict:
        """Collect filler statistics used across feedback sections."""
        filler_analysis = report_data.get('filler_analysis') or {}
        events = filler_analysis.get('acoustic_events') or []
        text_events = filler_analysis.get('text_model_fillers') or []
        
        label_counter: Counter = Counter()
        timestamps: List[float] = []
        
        def normalize_label(event: Dict) -> Optional[str]:
            label = (event.get('label') or event.get('token_original') or event.get('raw_label') or '').strip()
            return label or None
        
        for event in events + text_events:
            label = normalize_label(event)
            if label:
                label_counter[label] += 1
            start = event.get('start')
            if isinstance(start, (int, float)):
                timestamps.append(float(start))
        
        total_fillers = filler_analysis.get('total_fillers')
        if total_fillers is None:
            total_fillers = sum(label_counter.values())
        
        filler_ratio = filler_analysis.get('filler_ratio')
        if filler_ratio is None:
            total_words = report_data.get('total_words') or filler_analysis.get('total_words')
            if total_words:
                filler_ratio = total_fillers / total_words if total_words else 0
        
        top_label: Optional[str] = None
        top_count: int = 0
        if label_counter:
            top_label, top_count = label_counter.most_common(1)[0]
        
        timestamps_sorted = sorted(timestamps)[:3]
        
        return {
            "total_fillers": total_fillers,
            "filler_ratio": filler_ratio,
            "ratio_pct": (filler_ratio * 100) if isinstance(filler_ratio, (int, float)) else None,
            "top_label": top_label,
            "top_count": top_count,
            "timestamps": timestamps_sorted,
            "label_counter": label_counter
        }
    
    def _format_seconds(self, seconds: Optional[float]) -> str:
        if seconds is None or not isinstance(seconds, (int, float)):
            return "--:--"
        if seconds < 0:
            seconds = 0
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    
    def _build_filler_feedback(self, report_data: Dict) -> Dict:
        stats = self._extract_filler_stats(report_data)
        total_fillers = stats["total_fillers"]
        ratio_pct = stats["ratio_pct"]
        top_label = stats["top_label"]
        top_count = stats["top_count"]
        timestamps = stats["timestamps"]
        
        feedback: Dict[str, object] = {"growth": []}
        
        if ratio_pct is not None and ratio_pct <= 3.5:
            if total_fillers:
                detail = f"{total_fillers} filler instances ({ratio_pct:.1f}%)"
            else:
                detail = "minimal filler usage"
            highlight = f"You kept fillers under control with {detail}."
            if top_label:
                highlight += f" Most frequent sound was “{top_label}” only {top_count}x."
            feedback["strength"] = highlight
        else:
            if total_fillers:
                detail = f"{total_fillers} fillers (~{ratio_pct:.1f}% of words)"
            else:
                detail = "noticeable fillers"
            message = f"Detected {detail}"
            if top_label:
                message += f", mainly “{top_label}” ({top_count}×)"
            if timestamps:
                times = ", ".join(self._format_seconds(t) for t in timestamps)
                message += f" around {times}"
            message += ". Pause for a beat before answering and rehearse using the 4-7-8 breathing drill."
            feedback["growth"].append(message)
        
        return feedback
    
    def _build_pace_feedback(self, report_data: Dict) -> Dict:
        speaking_rate = report_data.get('speaking_rate_wpm')
        total_words = report_data.get('total_words')
        feedback: Dict[str, Optional[str]] = {}
        
        if not speaking_rate:
            return feedback
        
        if 135 <= speaking_rate <= 170:
            feedback["strength"] = f"Speaking pace stayed conversational at {speaking_rate:.0f} WPM."
        elif speaking_rate < 130:
            suggestion = f"You averaged {speaking_rate:.0f} WPM. Aim for 140–160 WPM by tightening explanations and practicing with a metronome script."
            if total_words:
                suggestion += f" Try trimming supporting detail so the {total_words} word script flows faster."
            feedback["growth"] = suggestion
        else:
            feedback["growth"] = f"You spoke quickly at {speaking_rate:.0f} WPM. Add intentional pauses and emphasize keywords to stay below 170 WPM."
        
        return feedback
    
    def _build_voice_feedback(self, report_data: Dict) -> Dict:
        voice_score = report_data.get('voice_confidence')
        speaking_metrics = report_data.get('speaking_metrics') or {}
        pause_summary = report_data.get('pause_summary') or {}
        total_pause_time = pause_summary.get('total_pause_time')
        longest_pause = pause_summary.get('longest_pause')
        
        feedback: Dict[str, Optional[str]] = {}
        
        if voice_score is None:
            return feedback
        
        if voice_score >= 75:
            strength = f"Voice confidence scored {voice_score:.0f}/100."
            if total_pause_time:
                strength += f" You balanced pauses well (total pause time {total_pause_time:.1f}s)."
            feedback["strength"] = strength
        else:
            message = f"Voice confidence landed at {voice_score:.0f}/100."
            if longest_pause and longest_pause > 1.5:
                message += f" Long pauses up to {longest_pause:.1f}s signaled hesitation."
            message += " Warm up with diaphragmatic breathing and resonance humming before recording."
            feedback["growth"] = message
        
        return feedback
    
    def _build_visual_feedback(self, report_data: Dict) -> Dict:
        eye_contact_raw = report_data.get('avg_eye_contact')
        facial_score = report_data.get('facial_confidence')
        dominant_emotion = report_data.get('dominant_emotion')
        
        eye_contact_pct = None
        if isinstance(eye_contact_raw, (int, float)):
            eye_contact_pct = eye_contact_raw * 100 if eye_contact_raw <= 1 else eye_contact_raw
        
        feedback: Dict[str, object] = {"growth": []}
        
        if facial_score is not None:
            if facial_score >= 75:
                feedback["primary"] = "Focused"
            elif facial_score >= 60:
                feedback["primary"] = "Reserved"
            elif facial_score >= 40:
                feedback["primary"] = "Tense"
            else:
                feedback["primary"] = "Static"
        else:
            feedback["primary"] = "Static"
        
        description_parts: List[str] = []
        if eye_contact_pct is not None:
            description_parts.append(f"Eye contact ~{eye_contact_pct:.0f}%")
        if dominant_emotion:
            description_parts.append(f"Dominant emotion: {dominant_emotion}")
        if facial_score is not None:
            description_parts.append(f"Facial confidence {facial_score:.0f}/100")
        if description_parts:
            feedback["description"] = "; ".join(description_parts)
        
        if eye_contact_pct is not None:
            if eye_contact_pct >= 55:
                feedback["strength"] = f"Maintained steady eye contact (~{eye_contact_pct:.0f}%)."
            elif eye_contact_pct < 45:
                feedback["growth"].append(
                    f"Eye contact averaged ~{eye_contact_pct:.0f}%. Practice the triangle gaze (left-center-right) every few sentences."
                )
        
        return feedback
    
    def _build_tone_feedback(self, report_data: Dict) -> Dict:
        voice_score = float(report_data.get('voice_confidence', 50) or 50)
        filler_stats = self._extract_filler_stats(report_data)
        ratio_pct = filler_stats.get("ratio_pct")
        speaking_rate = report_data.get('speaking_rate_wpm')
        tension_percentage = report_data.get('tension_percentage')
        tension_count = report_data.get('tension_count')
        emotion_timeline = report_data.get('emotion_timeline') or []
        dominant_emotion = report_data.get('dominant_emotion')

        effective_tension = None
        if isinstance(tension_percentage, (int, float)):
            effective_tension = max(0.0, min(1.0, tension_percentage))
        elif isinstance(tension_count, (int, float)) and isinstance(emotion_timeline, list) and emotion_timeline:
            effective_tension = max(0.0, min(1.0, tension_count / len(emotion_timeline)))

        emotion_counts: Counter = Counter()
        for frame in emotion_timeline:
            frame_emotion = frame.get('dominant_emotion') or frame.get('emotion')
            if isinstance(frame_emotion, str):
                emotion_counts[frame_emotion.lower()] += 1
        top_emotion: Optional[Tuple[str, float]] = None
        if emotion_counts:
            emotion_label, emotion_count = emotion_counts.most_common(1)[0]
            top_emotion = (emotion_label, emotion_count / sum(emotion_counts.values()))
            if not dominant_emotion:
                dominant_emotion = emotion_label

        if effective_tension is not None and effective_tension > 0.35:
            assessment = "Under Pressure"
        elif voice_score >= 78 and (effective_tension is None or effective_tension < 0.18):
            assessment = "Confident"
        elif voice_score >= 62 and (effective_tension is None or effective_tension < 0.28):
            assessment = "Calm"
        else:
            assessment = "Under Pressure"

        description_parts: List[str] = [f"Voice confidence {voice_score:.0f}/100"]
        if ratio_pct is not None:
            description_parts.append(f"Fillers {ratio_pct:.1f}%")
        if speaking_rate:
            description_parts.append(f"Pace {speaking_rate:.0f} WPM")
        if top_emotion:
            label, pct = top_emotion
            description_parts.append(f"DeepFace dominant {label} {pct*100:.0f}%")
        elif dominant_emotion:
            description_parts.append(f"DeepFace dominant {dominant_emotion}")
        if effective_tension is not None:
            description_parts.append(f"Tension {effective_tension*100:.0f}%")

        return {
            "assessment": assessment,
            "description": "; ".join(description_parts)
        }
    
    def _build_summary(
        self,
        report_data: Dict,
        filler_feedback: Dict,
        pace_feedback: Dict,
        visual_feedback: Dict
    ) -> List[str]:
        points: List[str] = []
        
        speaking_rate = report_data.get('speaking_rate_wpm')
        total_words = report_data.get('total_words')
        filler_stats = self._extract_filler_stats(report_data)
        ratio_pct = filler_stats.get("ratio_pct")
        top_label = filler_stats.get("top_label")
        
        if speaking_rate and ratio_pct is not None:
            line = f"Spoke at {speaking_rate:.0f} WPM with fillers at {ratio_pct:.1f}%"
            if top_label:
                line += f" (most common “{top_label}”)."
            points.append(line)
        
        eye_contact_raw = report_data.get('avg_eye_contact')
        if isinstance(eye_contact_raw, (int, float)):
            eye_pct = eye_contact_raw * 100 if eye_contact_raw <= 1 else eye_contact_raw
            dominant_emotion = report_data.get('dominant_emotion', 'neutral')
            points.append(f"Eye contact held ~{eye_pct:.0f}% with a {dominant_emotion} expression.")
        
        voice_score = report_data.get('voice_confidence')
        if voice_score is not None:
            points.append(f"Voice confidence scored {voice_score:.0f}/100; continue breath and projection drills.")
        
        if not points and total_words:
            points.append(f"Delivered {total_words} words with overall score {report_data.get('overall_score', 0):.1f}/100.")
        
        return points[:3]
    
    def _generate_fallback_feedback(self, report_data: Dict) -> Dict:
        """Generate minimal fallback feedback"""
        raw_feedback = {
            "strength": {"message": "Analysis completed successfully."},
            "growth_areas": ["Continue practising to improve communication skills."],
            "follow_up_questions": [
                {"question": "Can you summarise your key project outcome?", "timestamp": None}
            ],
            "metrics": self._build_metrics_summary(report_data),
            "pronunciation": {"summary": "Pronunciation analysis unavailable.", "issues": []},
            "summary": "Keep refining your delivery with focused practice."
        }
        coach_feedback = self._coerce_feedback_schema(raw_feedback, report_data, {}, None)
        coach_feedback.setdefault("pronunciation", {"summary": "Pronunciation analysis unavailable.", "issues": []})
        return coach_feedback
    
    def _prepare_context(self, report_data: Dict, transcript: str) -> str:
        """Prepare context string for OpenAI"""
        context_parts = []
        
        # Overall scores
        context_parts.append(f"Overall Score: {report_data.get('overall_score', 0)}/100")
        context_parts.append(f"Voice Confidence: {report_data.get('voice_confidence', 0)}/100")
        context_parts.append(f"Facial Confidence: {report_data.get('facial_confidence', 0)}/100")
        context_parts.append(f"Vocabulary Score: {report_data.get('vocabulary_score', 0)}/100")
        
        # Voice metrics (aggregated)
        filler_ratio = report_data.get('filler_word_ratio')
        if filler_ratio is None:
            filler_ratio = self._build_metrics_summary(report_data)["filler"]["percentage"] / 100 if report_data.get('total_words') else 0
        context_parts.append(f"\nVoice Metrics Summary:")
        context_parts.append(f"- Filler words: {report_data.get('filler_word_count', 0)} "
                             f"({float(filler_ratio)*100 if filler_ratio is not None else 0:.1f}% of words)")
        context_parts.append(f"- Speaking rate: {report_data.get('speaking_rate_wpm', 0)} WPM")
        context_parts.append(f"- Total pauses: {report_data.get('pause_count', 0)}")
        
        # Visual metrics (aggregated)
        if report_data.get('facial_confidence', 0) > 0:
            avg_eye = report_data.get('avg_eye_contact', 0)
            context_parts.append(f"\nVisual Metrics Summary:")
            context_parts.append(f"- Eye contact: {avg_eye*100:.0f}%")
        
        # Vocabulary metrics (aggregated)
        total_words = report_data.get('total_words', 0)
        context_parts.append(f"\nVocabulary Metrics Summary:")
        context_parts.append(f"- Total words: {total_words}")
        context_parts.append(f"- Unique words: {report_data.get('unique_words', 0)}")
        
        # Strengths and improvements (text only)
        strengths = report_data.get('strengths', [])
        improvements = report_data.get('improvements', [])
        if strengths:
            context_parts.append(f"\nTop strength noted: {strengths[0]}")
        if improvements:
            context_parts.append(f"Primary improvement: {improvements[0]}")
        
        return "\n".join(context_parts)

# Global instance
detailed_feedback_generator = DetailedFeedbackGenerator()

