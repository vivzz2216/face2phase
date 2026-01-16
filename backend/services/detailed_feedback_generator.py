"""
Detailed feedback generator for comprehensive presentation analysis
Generates Strength, Growth Area, Follow-up Questions, Tone, Visual Presence, Conciseness, Summary, and Pronunciation feedback
"""
import logging
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple
from .openai_enhancer import openai_enhancer
from ..exporters.report_generator import report_generator
from ..analysis.speech.pronunciation_analyzer import pronunciation_analyzer
from ..analysis.text.transcript_enhancer import transcript_enhancer
from ..analysis.text.vocabulary_enhancer import vocabulary_enhancer

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
            
            # CRITICAL: Extract transcript from report - this is the ACTUAL video/audio content
            transcript = report_data.get('transcript', '')
            enhanced_transcript = report_data.get('enhanced_transcript', transcript)
            words_with_timing = report_data.get('words_with_timing', [])
            
            # DEBUG: Log transcript extraction for verification
            logger.info(f"[TRANSCRIPT EXTRACTION] Session {session_id}:")
            logger.info(f"  - Transcript length: {len(transcript) if transcript else 0} chars")
            logger.info(f"  - Enhanced transcript length: {len(enhanced_transcript) if enhanced_transcript else 0} chars")
            logger.info(f"  - Words with timing: {len(words_with_timing)} words")
            if transcript:
                logger.info(f"  - Transcript preview: {transcript[:150]}..." if len(transcript) > 150 else f"  - Full transcript: {transcript}")
            else:
                logger.warning(f"  - ⚠️ NO TRANSCRIPT FOUND for session {session_id}")
            
            # DEBUG: Log voice confidence from report
            voice_confidence = report_data.get('voice_confidence', 0)
            logger.info(f"[VOICE CONFIDENCE] Session {session_id}: {voice_confidence}/100")
            if voice_confidence == 0:
                logger.warning(f"  - ⚠️ Voice confidence is 0 - check audio analysis pipeline")
                # Check if there's speaking_metrics with the score
                speaking_metrics = report_data.get('speaking_metrics', {})
                logger.info(f"  - Speaking metrics available: {bool(speaking_metrics)}")
                if speaking_metrics:
                    logger.info(f"  - Speaking rate: {speaking_metrics.get('speaking_rate_wpm', 'N/A')} WPM")
            
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
            
            # CRITICAL: Log what transcript is being sent to AI
            logger.info(f"[AI FEEDBACK] Sending to OpenAI:")
            logger.info(f"  - Transcript length: {len(enhanced_transcript)} chars")
            logger.info(f"  - Preview: {enhanced_transcript[:150]}...")
            
            prompt = f"""You are an expert presentation coach. Analyze this presentation and provide feedback.

IMPORTANT: Read the TRANSCRIPT below carefully. All questions MUST be about topics IN THIS TRANSCRIPT.

TRANSCRIPT TO ANALYZE:
\"\"\"
{enhanced_transcript[:3000]}
\"\"\"

ANALYSIS DATA:
{context}

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
                "word": "[word from transcript that was mispronounced]",
                "incorrect": "[how speaker said it]",
                "correct": "[correct pronunciation]",
                "phonetic": "[IPA]",
                "context": "[quote from transcript]"
            }}
        ],
        "general_advice": "Pronunciation advice based on the transcript"
    }}
}}

CRITICAL REQUIREMENTS FOR FOLLOW-UP QUESTIONS:
1. READ THE TRANSCRIPT CAREFULLY: Every question MUST reference specific content from the transcript above.
2. BE SPECIFIC: Extract actual topics, entities, technologies, projects, or concepts mentioned in the transcript.
   - Example: If transcript mentions "Python and MySQL", ask "Can you explain more about how you used Python with MySQL?"
   - Example: If transcript mentions "built a mobile app", ask "What was your development process for the mobile app?"
   - Example: If transcript mentions "faced challenges with deployment", ask "What challenges did you encounter during deployment?"
3. USE QUOTED CONTENT: If specific phrases are mentioned in quotes or emphasized, reference them in questions.
4. LINK TO MOMENTS: If possible, reference specific sentences or topics from different parts of the transcript.
5. NEVER USE GENERIC QUESTIONS: Avoid vague questions like "Can you tell us more?" without context.
   - BAD: "Can you elaborate?" 
   - GOOD: "Can you elaborate on the database architecture you mentioned?"
6. MINIMUM 3 QUESTIONS: Generate at least 3-5 questions, each about different specific topics from the transcript.

CRITICAL REQUIREMENTS FOR OTHER FIELDS:
2. STRENGTH: Based on what the speaker actually discussed in the transcript
3. SUMMARY: Summarize the actual content, topics, and themes from the transcript
4. PRONUNCIATION: Only analyze words that appear in the transcript above

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
            
            # CRITICAL: Always override AI-generated questions with our contextual questions
            # The AI sometimes generates generic questions despite instructions
            logger.info(f"[AI FEEDBACK] Overriding AI questions with contextual questions from transcript")
            
            # Remove markdown code blocks if present
            if feedback_text.startswith("```"):
                feedback_text = feedback_text.split("```")[1]
                if feedback_text.startswith("json"):
                    feedback_text = feedback_text[4:]
            
            raw_feedback = json.loads(feedback_text)
            
            # CRITICAL FIX: Always override AI-generated questions with our contextual questions
            # The AI sometimes generates generic questions about presentation skills (eye contact, etc.)
            # even when they don't match the actual video content. We must use transcript-based questions.
            logger.info(f"[AI FEEDBACK] Replacing AI questions with contextual transcript-based questions")
            contextual_questions = self._generate_contextual_questions(
                transcript,
                words_with_timing=report_data.get('words_with_timing', []),
                enhanced_transcript=enhanced_transcript,
                report_data=report_data
            )
            # Override AI questions with our contextual ones
            raw_feedback['follow_up_questions'] = contextual_questions
            logger.info(f"[AI FEEDBACK] Replaced with {len(contextual_questions)} contextual questions based on transcript")
            
            coach_feedback = self._coerce_feedback_schema(
                raw_feedback,
                report_data,
                pronunciation_analysis,
                pronunciation_score
            )
            # Attach supporting enhancements for UI components that still rely on them
            coach_feedback['transcript_improvement'] = transcript_improvement
            coach_feedback['vocabulary_enhancements'] = vocabulary_enhancements
            
            logger.info("AI feedback generated successfully with contextual transcript-based questions")
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
            
            # Follow-up Questions - Generate based on actual video/audio content
            follow_up_questions = self._generate_contextual_questions(
                transcript, 
                words_with_timing=report_data.get('words_with_timing', []),
                enhanced_transcript=report_data.get('enhanced_transcript', ''),
                report_data=report_data  # Pass full report data for audio/video analysis
            )
            
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
    
    def _generate_contextual_questions(
        self, 
        transcript: str, 
        words_with_timing: Optional[List[Dict]] = None,
        enhanced_transcript: Optional[str] = None,
        report_data: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Generate contextual follow-up questions based on actual video/audio content analysis
        
        Uses NLP, audio analysis, and video analysis to create questions specific to:
        - Topics and entities mentioned in transcript
        - Audio moments (pauses, pitch changes, filler words)
        - Video moments (facial expressions, emotions, eye contact)
        - Key insights from the analysis
        
        Args:
            transcript: Raw transcript text
            words_with_timing: Word-level timing info
            enhanced_transcript: Enhanced transcript with annotations
            report_data: Full report data with audio/video analysis
            
        Returns:
            List of contextual follow-up questions with timestamps
        """
        questions = []
        import re
        from collections import Counter
        
        # DEBUG: Log input for verification
        logger.info(f"[CONTEXTUAL QUESTIONS] Generating questions from video/audio content:")
        logger.info(f"  - Transcript length: {len(transcript) if transcript else 0} chars")
        logger.info(f"  - Words with timing: {len(words_with_timing) if words_with_timing else 0} entries")
        logger.info(f"  - Has report_data: {report_data is not None}")
        
        if not transcript or len(transcript.strip()) < 20:
            logger.warning(f"  - ⚠️ Empty/short transcript - transcript too short to generate contextual questions")
            logger.warning(f"  - Transcript value: {repr(transcript[:100]) if transcript else 'None'}")
            # Try to generate at least one question from whatever content exists
            if transcript and len(transcript.strip()) >= 10:
                # Very short transcript - generate minimal contextual question
                words = transcript.strip().split()[:10]
                if words:
                    meaningful_words = [w for w in words if len(w) > 2 and w.lower() not in ['uh', 'um', 'er', 'ah']]
                    if meaningful_words:
                        return [{"question": f"Can you share more details about {' '.join(meaningful_words[:5])}?", "timestamp": None}]
            # Return empty list - let the UI handle empty state
            return []
        
        # Log transcript preview (first 300 chars for better context)
        transcript_preview = transcript[:300].replace(chr(10), ' ').replace(chr(13), ' ')
        logger.info(f"  - Transcript preview: {transcript_preview}...")
        
        # Verify transcript has actual content (not just filler)
        transcript_words = transcript.split()
        if len(transcript_words) < 10:
            logger.warning(f"  - ⚠️ Transcript has very few words ({len(transcript_words)}) - questions may be generic")
        
        # Helper function to find timestamp for a word or phrase
        def get_timestamp_for_text(search_text: str, start_pos: int = 0) -> Optional[str]:
            """Find timestamp where text appears, starting from position"""
            if not words_with_timing:
                return None
            
            search_lower = search_text.lower()
            words_to_find = search_lower.split()
            if not words_to_find:
                return None
            
            # Search from start_pos onwards
            for i in range(start_pos, len(words_with_timing)):
                word_info = words_with_timing[i]
                word = word_info.get('word', '').lower()
                if words_to_find[0] in word:
                    # Check if full phrase matches (check next words)
                    match = True
                    for j, w in enumerate(words_to_find[1:], 1):
                        if i + j >= len(words_with_timing):
                            match = False
                            break
                        if w not in words_with_timing[i + j].get('word', '').lower():
                            match = False
                            break
                    if match:
                        time_sec = word_info.get('start', 0)
                        minutes = int(time_sec // 60)
                        seconds = int(time_sec % 60)
                        return f"{minutes}:{seconds:02d}"
            return None
        
        # Helper to get timestamp from seconds
        def format_timestamp(seconds: float) -> str:
            """Convert seconds to MM:SS format"""
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        
        # ===== STEP 1: Extract Topics and Entities from Transcript =====
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        words = transcript.lower().split()
        
        # Extract proper nouns and technical terms (more sophisticated)
        # Find capitalized terms (likely proper nouns, technologies, projects)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        technical_terms = re.findall(capitalized_pattern, transcript)
        # Remove common words that are always capitalized
        common_words = {'I', 'The', 'This', 'That', 'There', 'They', 'We', 'You', 'He', 'She', 'It', 'My', 'Today', 'Tomorrow', 'Yesterday', 'Now', 'Here', 'There'}
        # Also filter out names if they appear in common phrases (like "my name is")
        name_patterns = ['my name is', 'i am', 'i\'m', 'this is', 'i called', 'name\'s']
        technical_terms = [t for t in technical_terms if t not in common_words and len(t) > 2]
        
        # Filter out terms that appear right after "my name is" or similar phrases
        filtered_terms = []
        transcript_lower = transcript.lower()
        for term in technical_terms:
            term_lower = term.lower()
            # Check if this term appears after a name introduction phrase
            is_name = False
            for pattern in name_patterns:
                if pattern in transcript_lower:
                    pattern_pos = transcript_lower.find(pattern)
                    term_pos = transcript_lower.find(term_lower, pattern_pos)
                    # If term appears within 50 characters after the pattern, likely a name
                    if 0 <= term_pos <= pattern_pos + 50:
                        is_name = True
                        break
            if not is_name and term not in filtered_terms:
                filtered_terms.append(term)
        
        technical_terms = filtered_terms[:5]  # Limit to top 5
        
        # Extract quoted phrases (often important concepts)
        quoted_phrases = re.findall(r'"([^"]+)"', transcript)
        
        # Extract potential project/tool names (words followed by specific patterns)
        project_patterns = re.findall(r'\b([A-Z][a-z]+\s+(?:project|system|app|tool|framework|library|platform))\b', transcript, re.IGNORECASE)
        
        # Combine and deduplicate
        all_topics_raw = list(set(technical_terms + quoted_phrases + [p.split()[0] for p in project_patterns]))
        all_topics_raw = [t for t in all_topics_raw if len(t) > 2]
        
        # Filter out names and validate topics are meaningful
        # A good topic should either:
        # 1. Appear multiple times in transcript (not just once like a name)
        # 2. Be part of a compound term or technical phrase
        # 3. Appear in a technical context (near words like "system", "project", "tool", etc.)
        validated_topics = []
        transcript_lower_words = transcript_lower.split()
        
        for topic in all_topics_raw:
            topic_lower = topic.lower()
            topic_count = transcript_lower.count(topic_lower)
            
            # Skip if topic appears only once (likely a name or accidental mention)
            if topic_count < 2:
                # But allow if it's part of a technical phrase
                technical_context_words = ['system', 'project', 'tool', 'app', 'framework', 'library', 'platform', 'database', 'api', 'service']
                has_technical_context = any(ctx_word in transcript_lower[max(0, transcript_lower.find(topic_lower)-50):transcript_lower.find(topic_lower)+50] for ctx_word in technical_context_words)
                if not has_technical_context:
                    logger.info(f"  - Skipping topic '{topic}' - appears only once (likely a name)")
                    continue
            
            # Skip single-word topics that look like names (capitalized single word, rare in transcript)
            if len(topic.split()) == 1 and topic_count < 3:
                # Check if it's commonly a name pattern
                if topic[0].isupper() and len(topic) > 4 and topic_count < 3:
                    logger.info(f"  - Skipping topic '{topic}' - looks like a name (single word, rare)")
                    continue
            
            validated_topics.append(topic)
        
        all_topics = validated_topics[:5]  # Top 5 validated topics
        logger.info(f"  - Validated topics: {all_topics}")
        
        # Extract action verbs and their objects
        action_verbs = ['built', 'created', 'developed', 'designed', 'implemented', 'used', 
                       'made', 'tested', 'deployed', 'worked', 'studied', 'learned', 
                       'improved', 'optimized', 'analyzed', 'wrote', 'programmed', 'solved']
        actions_with_context = []
        for sentence in sentences[:10]:  # Check first 10 sentences
            sentence_lower = sentence.lower()
            for action in action_verbs:
                if action in sentence_lower:
                    # Extract object/noun after the verb
                    idx = sentence_lower.find(action)
                    remaining = sentence[idx + len(action):].strip()
                    # Get next few words as context
                    next_words = remaining.split()[:4]
                    if next_words:
                        actions_with_context.append({
                            'verb': action,
                            'context': ' '.join(next_words),
                            'sentence': sentence
                        })
                    break
        
        # ===== STEP 2: Analyze Audio Moments (if available) =====
        audio_moments = []
        if report_data:
            # Find significant pauses
            pauses = report_data.get('pauses_detailed', report_data.get('pauses', []))
            long_pauses = [p for p in pauses if p.get('duration', 0) > 2.0]  # Pauses > 2 seconds
            
            # Find moments with high filler word concentration
            filler_analysis = report_data.get('filler_analysis', {})
            filler_events = filler_analysis.get('acoustic_events', []) + \
                          filler_analysis.get('text_model_fillers', [])
            
            # Find pitch variations (indicating emotion/stress)
            pitch_data = report_data.get('pitch_data', {})
            pitch_timeline = pitch_data.get('pitch_timeline', [])
            
            if long_pauses:
                audio_moments.append({
                    'type': 'pause',
                    'time': long_pauses[0].get('start', 0),
                    'context': f"long pause ({long_pauses[0].get('duration', 0):.1f}s)"
                })
            
            if filler_events and len(filler_events) > 3:
                # Find cluster of fillers (indicates hesitation/stress point)
                filler_times = sorted([f.get('start', 0) for f in filler_events[:10]])
                for i, time in enumerate(filler_times[:-1]):
                    if filler_times[i+1] - time < 3.0:  # Multiple fillers within 3 seconds
                        audio_moments.append({
                            'type': 'filler_cluster',
                            'time': time,
                            'context': 'hesitation moment'
                        })
                        break
        
        # ===== STEP 3: Analyze Video Moments (if available) =====
        video_moments = []
        if report_data:
            facial_data = report_data.get('facial_analysis', {})
            emotion_timeline = facial_data.get('emotion_timeline', [])
            
            # Find high emotion moments (high confidence or strong emotion)
            if emotion_timeline:
                strong_emotions = [e for e in emotion_timeline 
                                 if e.get('confidence', 0) > 0.7 and 
                                 e.get('emotion') not in ['neutral', 'calm']]
                if strong_emotions:
                    video_moments.append({
                        'type': 'emotion',
                        'time': strong_emotions[0].get('timestamp', 0),
                        'emotion': strong_emotions[0].get('emotion', 'emotion'),
                        'context': f"strong {strong_emotions[0].get('emotion', 'emotion')} expression"
                    })
            
            # Find high eye contact moments
            high_eye_contact = report_data.get('high_eye_contact_moments', [])
            if high_eye_contact:
                video_moments.append({
                    'type': 'eye_contact',
                    'time': high_eye_contact[0].get('timestamp', 0),
                    'context': 'high eye contact moment'
                })
        
        # ===== STEP 4: Generate Contextual Questions =====
        
        # Question 1: About specific topic/entity mentioned (ENHANCED with filler word filtering)
        if all_topics:
            topic = all_topics[0]
            timestamp = get_timestamp_for_text(topic)
            # Extract more context from sentences mentioning this topic
            topic_sentences = [s for s in sentences if topic.lower() in s.lower()]
            if topic_sentences:
                # Find the sentence with most context
                best_sentence = max(topic_sentences, key=len)
                
                # Filter out filler words and clean the sentence
                filler_words = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically'}
                # Remove common words that don't add context
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
                
                # Extract key phrases before and after the topic
                words_in_sentence = best_sentence.split()
                try:
                    topic_idx = [i for i, w in enumerate(words_in_sentence) if topic.lower() in w.lower()][0]
                    # Get 2-4 words before and after topic for context
                    start_idx = max(0, topic_idx - 3)
                    end_idx = min(len(words_in_sentence), topic_idx + 4)
                    context_words = words_in_sentence[start_idx:end_idx]
                    
                    # Clean context: remove filler words and common words, keep meaningful terms
                    cleaned_context = []
                    for word in context_words:
                        word_clean = word.lower().strip('.,!?;:"()[]{}')
                        if word_clean and word_clean not in filler_words and word_clean not in common_words and len(word_clean) > 2:
                            # Keep original capitalization for proper nouns
                            cleaned_context.append(word.strip('.,!?;:"()[]{}'))
                    
                    # If we have cleaned context, use it; otherwise use a simpler phrase
                    if cleaned_context and len(cleaned_context) >= 2:
                        context_phrase = ' '.join(cleaned_context)  # Use all cleaned context words
                        # Create question with actual quote from transcript
                        # Extract a short quote (5-8 words) around the topic
                        quote_start = max(0, topic_idx - 2)
                        quote_end = min(len(words_in_sentence), topic_idx + 5)
                        quote_words = words_in_sentence[quote_start:quote_end]
                        # Filter quote to remove filler words
                        quote_cleaned = [w.strip('.,!?;:"()[]{}') for w in quote_words if w.lower().strip('.,!?;:"()[]{}') not in filler_words and len(w.strip('.,!?;:"()[]{}')) > 1]
                        if len(quote_cleaned) >= 3:
                            actual_quote = ' '.join(quote_cleaned[:6])  # Use first 6 words
                            question = f"When you said '{actual_quote}', you mentioned {topic}. Can you expand on what you meant by that and provide more context?"
                        else:
                            question = f"You mentioned {context_phrase} related to {topic}. Can you elaborate on what you meant by {topic} and how it relates to your work?"
                    elif len(words_in_sentence) > 5:
                        # Extract a quote from the sentence
                        quote_words = [w.strip('.,!?;:"()[]{}') for w in words_in_sentence[:8] if w.lower().strip('.,!?;:"()[]{}') not in filler_words and len(w.strip('.,!?;:"()[]{}')) > 1]
                        if len(quote_words) >= 4:
                            actual_quote = ' '.join(quote_words[:5])
                            question = f"You said '{actual_quote}...' Can you tell me more about that and what led you to discuss {topic}?"
                        else:
                            question = f"You discussed {topic} in your presentation. What specifically did you want your audience to understand about {topic}?"
                    else:
                        question = f"You mentioned {topic} during your presentation. Can you explain more about what {topic} means in the context of what you were discussing?"
                except (IndexError, ValueError):
                    # If extraction fails, use simpler question
                    question = f"You talked about {topic}. What was your experience with it?"
            else:
                question = f"I found {topic} interesting. Can you share more details about your experience with it?"
            
            questions.append({
                "question": question,
                "timestamp": timestamp
            })
            logger.info(f"  - Q1 about topic '{topic}' @ {timestamp}")
        
        # Question 2: About specific action/project mentioned (ENHANCED)
        if actions_with_context:
            action_info = actions_with_context[0]
            verb = action_info['verb']
            context = action_info['context']
            sentence = action_info['sentence']
            timestamp = get_timestamp_for_text(verb)
            
            # Extract actual quote from the sentence with context
            sentence_words = sentence.split()
            filler_words = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically'}
            
            # Find verb position and extract quote around it
            try:
                verb_idx = next(i for i, w in enumerate(sentence_words) if verb.lower() in w.lower())
                start = max(0, verb_idx - 1)
                end = min(len(sentence_words), verb_idx + 6)
                quote_words_raw = sentence_words[start:end]
                # Clean quote - remove fillers
                quote_words = [w.strip('.,!?;:"()[]{}') for w in quote_words_raw if w.lower().strip('.,!?;:"()[]{}') not in filler_words and len(w.strip('.,!?;:"()[]{}')) > 0]
                actual_quote = ' '.join(quote_words[:7])  # Use up to 7 words for quote
                
                # Create question with actual quote from transcript
                if verb in ['built', 'created', 'developed', 'made']:
                    question = f"You said '{actual_quote}'. Can you walk me through the process you used and what challenges you faced along the way?"
                elif verb in ['solved', 'improved', 'optimized', 'fixed']:
                    question = f"When you mentioned '{actual_quote}', what specific approach did you take and what results did you achieve?"
                elif verb in ['learned', 'studied', 'researched']:
                    question = f"You said '{actual_quote}'. What were the most important insights you gained from that experience?"
                elif verb in ['used', 'implemented', 'applied']:
                    question = f"Regarding '{actual_quote}', how did that work out for you and why did you choose that particular approach?"
                else:
                    question = f"You mentioned '{actual_quote}'. Can you provide more details about what you meant and share more context?"
            except (StopIteration, ValueError):
                # Fallback if verb not found - use sentence start
                quote_words = [w.strip('.,!?;:"()[]{}') for w in sentence_words[:6] if w.lower().strip('.,!?;:"()[]{}') not in filler_words]
                if quote_words:
                    actual_quote = ' '.join(quote_words[:5])
                    question = f"You said '{actual_quote}...' Can you expand on that and explain more about {verb} {context}?"
                else:
                    question = f"You mentioned {verb} {context}. Can you elaborate on that and provide more details?"
            
            questions.append({
                "question": question,
                "timestamp": timestamp
            })
            logger.info(f"  - Q2 about action '{verb}' @ {timestamp}")
        
        # Question 3: About audio moment (pause, hesitation, pitch change) - ENHANCED
        if audio_moments:
            moment = audio_moments[0]
            timestamp = format_timestamp(moment['time'])
            
            if moment['type'] == 'pause':
                # Find what was said just before and after the pause for better context
                if words_with_timing:
                    pause_idx = next((i for i, w in enumerate(words_with_timing) 
                                    if w.get('start', 0) >= moment['time'] - 0.5), None)
                    if pause_idx and pause_idx > 0:
                        # Get words before pause
                        prev_words = [w.get('word', '') for w in words_with_timing[max(0, pause_idx-4):pause_idx]]
                        # Get words after pause (if available)
                        next_words = []
                        if pause_idx < len(words_with_timing) - 1:
                            next_words = [w.get('word', '') for w in words_with_timing[pause_idx:min(len(words_with_timing), pause_idx+3)]]
                        
                        context_before = ' '.join(prev_words)
                        if next_words:
                            context_after = ' '.join(next_words)
                            question = f"After you said '{context_before}', there was a thoughtful pause before you continued with '{context_after}'. What consideration went into that moment?"
                        else:
                            question = f"There was a {moment['context']} after '{context_before}'. What were you reflecting on or deciding at that point?"
                    else:
                        question = f"You had a {moment['context']} during your presentation. Was there a particular point you wanted to emphasize or consider more carefully?"
                else:
                    question = f"You had a {moment['context']} during your presentation. What was going through your mind at that moment?"
            elif moment['type'] == 'filler_cluster':
                # Find context around filler cluster
                if words_with_timing:
                    filler_idx = next((i for i, w in enumerate(words_with_timing) 
                                     if abs(w.get('start', 0) - moment['time']) < 1.0), None)
                    if filler_idx is not None:
                        context_words = [w.get('word', '') for w in words_with_timing[max(0, filler_idx-3):min(len(words_with_timing), filler_idx+4)]]
                        context_phrase = ' '.join(context_words)
                        question = f"There was some hesitation when discussing '{context_phrase}'. What made that part more challenging, and how did you work through it?"
                    else:
                        question = f"I noticed some hesitation during that {moment['context']}. What aspect was more difficult to articulate?"
                else:
                    question = f"There was noticeable hesitation at that {moment['context']}. What made that part of your presentation challenging?"
            else:
                question = f"At that {moment['context']}, what were you trying to emphasize?"
            
            questions.append({
                "question": question,
                "timestamp": timestamp
            })
            logger.info(f"  - Q3 about audio moment @ {timestamp}")
        
        # Question 4: About video moment (emotion, expression, eye contact)
        if video_moments and len(questions) < 4:
            moment = video_moments[0]
            timestamp = format_timestamp(moment['time'])
            
            if moment['type'] == 'emotion':
                emotion = moment.get('emotion', 'emotion')
                # Find what was said at that moment
                if words_with_timing:
                    emotion_idx = next((i for i, w in enumerate(words_with_timing) 
                                      if abs(w.get('start', 0) - moment['time']) < 2.0), None)
                    if emotion_idx is not None:
                        context_words = [w.get('word', '') for w in words_with_timing[max(0, emotion_idx-2):min(len(words_with_timing), emotion_idx+3)]]
                        context_phrase = ' '.join(context_words)
                        question = f"When you said '{context_phrase}', I noticed a {emotion} expression. What emotion were you experiencing?"
                    else:
                        question = f"I noticed a {emotion} expression during your presentation. What were you feeling at that moment?"
                else:
                    question = f"You showed a {emotion} expression. What was that moment about?"
            elif moment['type'] == 'eye_contact':
                question = f"During that {moment['context']}, you seemed very engaged. What were you focusing on?"
            else:
                question = f"At that {moment['context']}, what was going through your mind?"
            
            questions.append({
                "question": question,
                "timestamp": timestamp
            })
            logger.info(f"  - Q4 about video moment @ {timestamp}")
        
        # Question 5: About challenges or outcomes (from transcript analysis)
        challenge_words = ['challenge', 'problem', 'difficult', 'issue', 'struggle', 'hard', 'complex']
        outcome_words = ['result', 'outcome', 'achieved', 'success', 'completed', 'improvement', 'benefit']
        
        has_challenges = any(w in transcript.lower() for w in challenge_words)
        has_outcomes = any(w in transcript.lower() for w in outcome_words)
        
        if has_challenges and len(questions) < 5:
            challenge_word = next((w for w in challenge_words if w in transcript.lower()), None)
            if challenge_word:
                timestamp = get_timestamp_for_text(challenge_word)
                # Find the sentence with challenge and extract more context
                challenge_sentences = [s for s in sentences if challenge_word in s.lower()]
                if challenge_sentences:
                    challenge_sentence = challenge_sentences[0]
                    # Extract 8-12 words around the challenge word for context
                    words = challenge_sentence.split()
                    try:
                        challenge_idx = next(i for i, w in enumerate(words) if challenge_word.lower() in w.lower())
                        start = max(0, challenge_idx - 2)
                        end = min(len(words), challenge_idx + 6)
                        quote_words = [w.strip('.,!?;:"()[]{}') for w in words[start:end] if len(w.strip('.,!?;:"()[]{}')) > 0]
                        actual_quote = ' '.join(quote_words)
                        # Clean quote
                        filler_words = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well'}
                        quote_cleaned = [w for w in quote_words if w.lower() not in filler_words]
                        if len(quote_cleaned) >= 3:
                            clean_quote = ' '.join(quote_cleaned[:5])
                            question = f"You said '{clean_quote}'. Can you describe the specific challenges you faced and how you addressed them?"
                        else:
                            question = f"You mentioned encountering challenges. Can you explain what those challenges were and what strategies you used to overcome them?"
                    except (StopIteration, ValueError):
                        # Extract quote from sentence start
                        quote_words = [w.strip('.,!?;:"()[]{}') for w in challenge_sentence.split()[:8] if len(w.strip('.,!?;:"()[]{}')) > 1]
                        if len(quote_words) >= 4:
                            actual_quote = ' '.join(quote_words[:5])
                            question = f"When you said '{actual_quote}...', you mentioned facing challenges. What were those challenges and how did you work through them?"
                        else:
                            question = f"You talked about facing challenges. Can you share more details about what obstacles you encountered and how you navigated them?"
                else:
                    question = "You mentioned facing some challenges. Can you walk us through what obstacles you encountered and how you navigated them?"
                
                questions.append({
                    "question": question,
                    "timestamp": timestamp
                })
                logger.info(f"  - Q5 about challenges @ {timestamp}")
        
        elif has_outcomes and len(questions) < 5:
            outcome_word = next((w for w in outcome_words if w in transcript.lower()), None)
            if outcome_word:
                timestamp = get_timestamp_for_text(outcome_word)
                outcome_sentences = [s for s in sentences if outcome_word in s.lower()]
                if outcome_sentences:
                    outcome_sentence = outcome_sentences[0]
                    # Extract context around outcome
                    words = outcome_sentence.split()
                    try:
                        outcome_idx = next(i for i, w in enumerate(words) if outcome_word.lower() in w.lower())
                        start = max(0, outcome_idx - 2)
                        end = min(len(words), outcome_idx + 6)
                        quote_words = [w.strip('.,!?;:"()[]{}') for w in words[start:end] if len(w.strip('.,!?;:"()[]{}')) > 0]
                        filler_words = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well'}
                        quote_cleaned = [w for w in quote_words if w.lower() not in filler_words]
                        if len(quote_cleaned) >= 3:
                            actual_quote = ' '.join(quote_cleaned[:5])
                            question = f"You mentioned '{actual_quote}'. What was the significance of that outcome and how did it impact your overall goals?"
                        else:
                            question = f"You discussed achieving results. Can you explain what those outcomes were and why they were important?"
                    except (StopIteration, ValueError):
                        # Extract quote from sentence
                        quote_words = [w.strip('.,!?;:"()[]{}') for w in outcome_sentence.split()[:7] if len(w.strip('.,!?;:"()[]{}')) > 1]
                        if len(quote_words) >= 4:
                            actual_quote = ' '.join(quote_words[:4])
                            question = f"When you said '{actual_quote}...', you mentioned outcomes. What were those results and what did you learn from them?"
                        else:
                            question = f"You discussed some outcomes. What were the most meaningful results you achieved, and why were they important?"
                else:
                    question = "You discussed some outcomes. What were the most meaningful results you achieved, and why were they important?"
                
                questions.append({
                    "question": question,
                    "timestamp": timestamp
                })
                logger.info(f"  - Q5 about outcomes @ {timestamp}")
        
        # Question 6: About future plans or next steps (ENHANCED)
        if len(questions) < 5:
            future_words = ['future', 'next', 'plan', 'will', 'going to', 'intend', 'hope', 'improve', 'enhance', 'expand']
            # Check last 30% of transcript for future references (usually mentioned near end)
            transcript_end = transcript[-int(len(transcript) * 0.3):] if len(transcript) > 100 else transcript
            future_mentions = [w for w in future_words if w in transcript_end.lower()]
            
            if future_mentions:
                future_word = future_mentions[0]
                timestamp = get_timestamp_for_text(future_word, max(0, len(words_with_timing) - 50) if words_with_timing else 0)
                # Find sentence with future word
                future_sentences = [s for s in sentences if future_word.lower() in s.lower() and s in transcript_end]
                if future_sentences:
                    future_sentence = future_sentences[-1]  # Get last one (most recent)
                    # Extract quote from sentence
                    words = future_sentence.split()
                    filler_words = {'uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well'}
                    quote_words = [w.strip('.,!?;:"()[]{}') for w in words[:8] if w.lower().strip('.,!?;:"()[]{}') not in filler_words and len(w.strip('.,!?;:"()[]{}')) > 1]
                    if len(quote_words) >= 4:
                        actual_quote = ' '.join(quote_words[:5])
                        question = f"You said '{actual_quote}...' Can you elaborate on what concrete steps you're planning to take next and what outcomes you're hoping to achieve?"
                    else:
                        question = f"You touched on future directions. Can you explain what you're planning to do next and what you're most excited about pursuing?"
                else:
                    question = "You touched on future directions. Can you elaborate on your immediate next steps and what you're most excited about pursuing?"
            else:
                # Use last sentence timestamp - ask about next steps even if not explicitly mentioned
                timestamp = None
                if words_with_timing and len(words_with_timing) > 0:
                    last_word = words_with_timing[-1]
                    timestamp = format_timestamp(last_word.get('start', 0))
                # More engaging question even without explicit future mention
                question = "Looking forward, what are you planning to focus on next with this work? Are there areas you'd like to explore further?"
            
            questions.append({
                "question": question,
                "timestamp": timestamp
            })
            logger.info(f"  - Q6 about future plans @ {timestamp}")
        
        # Ensure we have questions if possible, but don't force generic dummy questions
        # Only add more questions if we have transcript context to work with
        if len(questions) < 3 and transcript and len(transcript.strip()) > 50:
            # Try to generate one more contextual question using transcript content
            mid_timestamp = None
            if words_with_timing and len(words_with_timing) > 10:
                mid_point = len(words_with_timing) // 2
                mid_timestamp = format_timestamp(words_with_timing[mid_point].get('start', 0))
                mid_words = [w.get('word', '') for w in words_with_timing[mid_point-2:mid_point+3] if w.get('word')]
                # Filter out filler words
                meaningful_words = [w for w in mid_words if w.lower() not in ['uh', 'um', 'er', 'ah', 'like', 'you know', 'so', 'well'] and len(w) > 2]
                if meaningful_words:
                    question = f"Can you provide more details about {' '.join(meaningful_words)}?"
                    questions.append({
                        "question": question,
                        "timestamp": mid_timestamp
                    })
        
        # Filter out generic presentation skill questions that don't reference transcript content
        # These are questions about general presentation skills, not about the actual content
        generic_patterns = [
            'maintain.*eye contact', 'strategies.*eye contact', 'how.*maintain',
            'what.*strategy', 'specific strategies', 'tips.*improve',
            'how can you', 'what.*technique', 'practice.*presentation'
        ]
        
        # Only filter if transcript has actual content
        transcript_has_content = transcript and len(transcript.strip()) > 50
        
        # Deduplicate questions and filter generic ones
        unique_questions = []
        seen_texts = set()
        import re
        for q in questions[:10]:  # Check more questions to have good ones after filtering
            q_text = q['question'].lower().strip()
            
            # Skip generic presentation skill questions that don't reference transcript
            is_generic = False
            if transcript_has_content:
                for pattern in generic_patterns:
                    if re.search(pattern, q_text):
                        # Check if question references any actual content from transcript
                        transcript_words = set(transcript.lower().split())
                        q_words = set(q_text.split())
                        # If less than 10% overlap with transcript, it's too generic
                        if len(q_words & transcript_words) / max(len(q_words), 1) < 0.1:
                            is_generic = True
                            logger.info(f"  - Filtered generic question: {q['question'][:60]}...")
                            break
            
            if is_generic:
                continue
            
            # Check if question is too similar to existing ones
            is_duplicate = False
            for seen in seen_texts:
                # If more than 70% words overlap, consider duplicate
                q_words = set(q_text.split())
                seen_words = set(seen.split())
                if q_words and len(q_words & seen_words) / len(q_words) > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_questions.append(q)
                seen_texts.add(q_text)
                if len(unique_questions) >= 5:
                    break
        
        # DEBUG: Log generated questions with transcript context verification
        logger.info(f"  - Generated {len(unique_questions)} contextual questions:")
        for i, q in enumerate(unique_questions, 1):
            q_text = q['question']
            # Verify question references transcript content
            q_words = set(q_text.lower().split())
            transcript_words_lower = set(transcript.lower().split())
            overlap = len(q_words & transcript_words_lower)
            overlap_pct = (overlap / len(q_words) * 100) if q_words else 0
            logger.info(f"    Q{i}: {q_text[:70]}... @ {q.get('timestamp', 'N/A')} (transcript overlap: {overlap} words, {overlap_pct:.1f}%)")
        
        if len(unique_questions) < 1:
            logger.warning(f"  - ⚠️ No questions generated - transcript may lack sufficient content")
            # Try to generate at least one question from transcript even if validation failed
            if transcript and len(transcript.strip()) > 30:
                sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
                if sentences:
                    first_sentence = sentences[0]
                    words = first_sentence.split()[:8]
                    meaningful = [w for w in words if len(w.strip('.,!?;:"()[]{}')) > 2 and w.lower() not in ['uh', 'um', 'er', 'ah', 'like', 'you know']]
                    if meaningful:
                        timestamp = None
                        if words_with_timing and len(words_with_timing) > 0:
                            timestamp = format_timestamp(words_with_timing[0].get('start', 0))
                        unique_questions.append({
                            "question": f"Can you elaborate on {' '.join(meaningful[:6])}?",
                            "timestamp": timestamp
                        })
        
        if len(unique_questions) < 3:
            logger.warning(f"  - ⚠️ Only {len(unique_questions)} questions generated - may need more transcript content")
        
        # Ensure we return at least one question if transcript has content
        if len(unique_questions) == 0 and transcript and len(transcript.strip()) > 20:
            logger.warning(f"  - ⚠️ Fallback: Generating minimal question from transcript")
            words = transcript.strip().split()[:12]
            meaningful = [w.strip('.,!?;:"()[]{}') for w in words if len(w.strip('.,!?;:"()[]{}')) > 2]
            if meaningful:
                unique_questions.append({
                    "question": f"What would you like to expand on regarding {' '.join(meaningful[:8])}?",
                    "timestamp": None
                })
        
        return unique_questions[:5]  # Return max 5 questions
    
    def _generate_fallback_feedback(self, report_data: Dict) -> Dict:
        """Generate minimal fallback feedback"""
        raw_feedback = {
            "strength": {"message": "Analysis completed successfully."},
            "growth_areas": ["Continue practising to improve communication skills."],
            "follow_up_questions": [],  # Empty - no dummy questions. Real questions generated from transcript.
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

