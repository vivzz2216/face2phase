"""
Chatbot service for interactive video analysis guidance
Uses OpenAI API to provide context-aware responses about the video/analysis
"""
import logging
from typing import Dict, List, Optional, Iterable, Generator
from datetime import timedelta
from .openai_enhancer import openai_enhancer

logger = logging.getLogger(__name__)

class ChatbotService:
    """Handles chatbot conversations with video analysis context"""
    
    def __init__(self):
        self.openai = openai_enhancer
        # Store conversation history per session
        self.conversations = {}
        # Store pre-computed session insights/context
        self.session_context = {}
    
    def get_response(self, session_id: str, user_message: str, report_data: Dict) -> str:
        """
        Get chatbot response for user message
        
        Args:
            session_id: Session identifier
            user_message: User's message/question
            report_data: Complete report data for context
            
        Returns:
            Chatbot response string
        """
        try:
            # CRITICAL: Check if question is related to the video/audio content
            if not self._is_question_relevant(user_message, report_data):
                logger.info(f"[CHATBOT] Blocked unrelated question: {user_message[:100]}")
                return ("I can only answer questions about your presentation/video analysis. "
                       "Please ask about your scores, speaking patterns, facial expressions, "
                       "transcript content, or how to improve your communication skills.")
            
            messages = self._prepare_conversation(session_id, user_message, report_data)

            if self.openai.enabled:
                response = self._get_ai_response(messages)
            else:
                response = self._get_fallback_response(user_message, report_data)

            self._store_assistant_turn(session_id, response)

            return response

        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def _is_question_relevant(self, user_message: str, report_data: Dict) -> bool:
        """Check if question relates to video content - blocks general knowledge queries"""
        message_lower = user_message.lower().strip()
        
        # Allowed keywords (presentation/video/analysis-related) - check FIRST
        allowed = {
            'score', 'scores', 'voice', 'speaking', 'speech', 'facial', 'face', 'transcript',
            'improve', 'improvement', 'feedback', 'my', 'presentation', 'video', 'audio',
            'analysis', 'analytics', 'eye', 'contact', 'pacing', 'pace', 'filler', 'fillers',
            'how', 'why', 'when', 'where', 'can', 'should', 'would', 'could',
            'confidence', 'performance', 'delivery', 'tone', 'emotion', 'gesture', 'gestures',
            'body language', 'posture', 'timing', 'pauses', 'pause', 'words', 'vocabulary',
            'pronunciation', 'clarity', 'rate', 'speed', 'recommendation', 'suggestion',
            'insight', 'insights', 'metric', 'metrics', 'result', 'results', 'statistic'
        }
        
        # Check if question contains allowed keywords FIRST
        has_allowed_keyword = any(keyword in message_lower for keyword in allowed)
        
        # Also check if question is clearly about "my" presentation/video (first person)
        is_personal = any(phrase in message_lower for phrase in [
            'my presentation', 'my video', 'my speaking', 'my performance', 'my score',
            'my voice', 'my analysis', 'my feedback', 'about me', 'my delivery'
        ])
        
        # If question has allowed keywords or is personal, allow it (even if it contains "what is")
        if has_allowed_keyword or is_personal:
            return True
        
        # Now check for blocked phrases (only if no allowed keywords found)
        blocked_phrases = [
            'who is', 'who are', 'who was', 'who were',
            'what is', 'what are', 'what was', 'what were', 
            'tell me about', 'which animal', 'what animal',
            'capital of', 'world war',
        ]
        
        # Check for blocked phrases
        for phrase in blocked_phrases:
            if phrase in message_lower:
                logger.info(f"[CHATBOT] Blocked irrelevant question (contains phrase '{phrase}'): {message_lower[:100]}")
                return False
        
        # Blocked single words (celebrities, sports, general knowledge)
        blocked_words = {
            'messi', 'ronaldo', 'neymar', 'mbappe', 'football', 'soccer', 'cricket', 'basketball', 
            'tennis', 'golf', 'baseball', 'hockey', 'rugby',
            'celebrity', 'actor', 'actress', 'president', 'politics', 'capital', 'movie', 'film',
            'animal', 'fur', 'famous', 'history', 'country', 'city', 'continent',
            'weather', 'temperature', 'recipe', 'cooking', 'food', 'restaurant', 'music', 'song',
            'sport', 'team', 'player', 'game', 'match', 'tournament', 'league', 'championship',
            'planet', 'star', 'galaxy', 'universe', 'math', 'mathematics', 'physics', 'chemistry',
            'biology', 'geography', 'economics', 'philosophy'
        }
        
        # Check for blocked words
        message_words = set(message_lower.split())
        blocked_found = message_words & blocked_words
        if blocked_found:
            logger.info(f"[CHATBOT] Blocked irrelevant question (contains words {blocked_found}): {message_lower[:100]}")
            return False
        
        # If we get here and no allowed keywords found, block it
        logger.info(f"[CHATBOT] Blocked - no relevant keywords found: {message_lower[:100]}")
        return False
    

    def stream_response(self, session_id: str, user_message: str, report_data: Dict) -> Iterable[str]:
        """
        Stream chatbot response chunks (for real-time output).
        """
        try:
            # CRITICAL: Check if question is related to the video/audio content BEFORE processing
            if not self._is_question_relevant(user_message, report_data):
                logger.info(f"[CHATBOT] Blocked unrelated question (stream): {user_message[:100]}")
                blocked_message = ("I can only answer questions about your presentation/video analysis. "
                                 "Please ask about your scores, speaking patterns, facial expressions, "
                                 "transcript content, or how to improve your communication skills.")
                self._store_assistant_turn(session_id, blocked_message)
                yield blocked_message
                return
            
            messages = self._prepare_conversation(session_id, user_message, report_data)

            if self.openai.enabled:
                final_text = []
                for chunk in self._get_ai_response_stream(messages):
                    if chunk:
                        final_text.append(chunk)
                        yield chunk
                response_text = "".join(final_text).strip()
                if response_text:
                    self._store_assistant_turn(session_id, response_text)
            else:
                response_text = self._get_fallback_response(user_message, report_data)
                self._store_assistant_turn(session_id, response_text)
                yield response_text
        except Exception as exc:
            logger.error(f"Streaming chatbot error: {exc}")
            error_message = "I'm having trouble processing that right now. Please retry in a moment."
            self._store_assistant_turn(session_id, error_message)
            yield error_message

    def _prepare_conversation(self, session_id: str, user_message: str, report_data: Dict) -> List[Dict]:
        """Initialise conversation history, add user message, and build message payload."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            "role": "user",
            "content": user_message
        })

        system_context = self._prepare_system_context(report_data)

        messages = [{"role": "system", "content": system_context}]
        messages.extend(self.conversations[session_id][-12:])  # maintain last turns
        return messages

    def _store_assistant_turn(self, session_id: str, response: str):
        """Persist assistant turn back into history."""
        self.conversations.setdefault(session_id, []).append({
            "role": "assistant",
            "content": response
        })

    def _get_ai_response(self, messages: List[Dict]) -> str:
        """Get response from OpenAI API - concise and precise"""
        try:
            response = self.openai.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,  # Reduced for concise responses
                temperature=0.5  # Lower for more precise/focused responses
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error in chatbot: {e}")
            return "I'm having trouble connecting to the AI service. Please try again later."
    
    def _get_ai_response_stream(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Yield streaming chunks from OpenAI API - concise mode."""
        try:
            stream = self.openai.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,  # Reduced for concise responses
                temperature=0.5,  # Lower for more precise/focused responses
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:
            logger.error(f"OpenAI streaming error: {exc}")
            raise

    def _get_fallback_response(self, user_message: str, report_data: Dict) -> str:
        """Generate fallback response without AI"""
        user_lower = user_message.lower()
        
        # Check for common questions
        if any(word in user_lower for word in ['score', 'rating', 'overall']):
            score = report_data.get('overall_score', 0)
            return f"Your overall communication score is {score}/100. "
        
        elif any(word in user_lower for word in ['voice', 'speaking', 'audio']):
            voice_score = report_data.get('voice_confidence', 0)
            filler_count = report_data.get('filler_word_count', 0)
            return f"Your voice confidence score is {voice_score}/100. You used {filler_count} filler words. "
        
        elif any(word in user_lower for word in ['facial', 'face', 'eye contact', 'visual']):
            facial_score = report_data.get('facial_confidence', 0)
            eye_contact = report_data.get('avg_eye_contact', 0)
            return f"Your facial confidence score is {facial_score}/100. Average eye contact: {eye_contact:.1%}. "
        
        elif any(word in user_lower for word in ['vocabulary', 'words', 'language']):
            vocab_score = report_data.get('vocabulary_score', 0)
            unique_words = report_data.get('unique_words', 0)
            return f"Your vocabulary score is {vocab_score}/100. You used {unique_words} unique words. "
        
        elif any(word in user_lower for word in ['improve', 'better', 'suggestion', 'tip']):
            improvements = report_data.get('improvements', [])
            if improvements:
                return f"Here are some suggestions: {improvements[0]}. "
            return "Continue practicing to improve your communication skills. "
        
        elif any(word in user_lower for word in ['transcript', 'what said', 'said']):
            transcript = report_data.get('transcript', '')
            if transcript:
                preview = transcript[:200] + "..." if len(transcript) > 200 else transcript
                return f"Here's a preview of your transcript: {preview}"
            return "Transcript is not available for this session. "
        
        else:
            return "I can help you understand your presentation analysis. Ask me about scores, voice, facial expressions, vocabulary, or improvements. "
    
    def _prepare_system_context(self, report_data: Dict) -> str:
        """Prepare system context for chatbot - optimized for concise responses"""
        context_parts = [
            "You are an expert presentation coach. Be CONCISE and PRECISE.",
            "",
            "CRITICAL RULES:",
            "- ONLY answer questions about THIS user's presentation/video analysis",
            "- NEVER answer general knowledge questions (sports, celebrities, history, etc.)",
            "- If asked about anything unrelated to the presentation, say: 'I can only answer questions about your presentation/video analysis.'",
            "- Keep responses under 3-4 sentences maximum",
            "- Give actionable advice, not lengthy explanations",
            "- Use bullet points for multiple items",
            "- Reference specific metrics when relevant",
            "- No filler phrases like 'Great question!' or 'I'd be happy to help'",
            "",
            "SCORES:"
        ]
        
        # Key metrics only
        context_parts.append(f"Overall: {report_data.get('overall_score', 0)}/100")
        context_parts.append(f"Voice: {report_data.get('voice_confidence', 0)}/100")
        context_parts.append(f"Facial: {report_data.get('facial_confidence', 0)}/100")
        context_parts.append(f"Vocab: {report_data.get('vocabulary_score', 0)}/100")
        
        # Filler info
        filler_count = report_data.get('filler_word_count', 0)
        if filler_count:
            context_parts.append(f"Fillers: {filler_count}")
        
        # Transcript preview (shorter)
        transcript = report_data.get('transcript', '')
        if transcript:
            context_parts.append(f"\nTRANSCRIPT (first 500 chars):\n{transcript[:500]}")
        
        # Top strengths and improvements (just 2 each)
        strengths = report_data.get('strengths', [])
        improvements = report_data.get('improvements', [])
        
        if strengths:
            context_parts.append(f"\nSTRENGTHS: {'; '.join(strengths[:2])}")
        
        if improvements:
            context_parts.append(f"\nNEEDS WORK: {'; '.join(improvements[:2])}")
        
        return "\n".join(context_parts)
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.session_context:
            del self.session_context[session_id]

    def get_session_context(self, session_id: str, report_data: Dict) -> Dict:
        """Return cached session insights for quick prompts/context chips."""
        if session_id not in self.session_context:
            self.session_context[session_id] = self._build_session_context(report_data)
        return self.session_context[session_id]

    def _build_session_context(self, report_data: Dict) -> Dict:
        """Derive quick insights, flagged moments, and quick prompts from report."""
        insights = []
        flags = []
        quick_prompts = []

        def fmt_pct(value, default="--"):
            if value is None:
                return default
            try:
                return f"{float(value):.0f}%"
            except Exception:
                return default

        def format_time(seconds: float) -> str:
            if seconds is None:
                return "--:--"
            try:
                td = timedelta(seconds=float(seconds))
                total_seconds = int(td.total_seconds())
                minutes, secs = divmod(total_seconds, 60)
                return f"{minutes}:{secs:02d}"
            except Exception:
                return "--:--"

        overall = report_data.get('overall_score')
        if overall is not None:
            insights.append({
                "id": "overall",
                "label": "Overall Score",
                "value": f"{overall:.1f}/100",
                "tone": "primary"
            })

        voice = report_data.get('voice_confidence')
        filler_count = report_data.get('filler_word_count')
        if voice is not None:
            insights.append({
                "id": "voice",
                "label": "Voice Confidence",
                "value": f"{voice:.1f}/100",
                "hint": f"{filler_count or 0} fillers detected"
            })

        visual_metrics = report_data.get('visual_analytics', {}).get('tension_summary', {})
        if visual_metrics:
            insights.append({
                "id": "visual",
                "label": "Eye Contact",
                "value": fmt_pct(visual_metrics.get('avg_eye_contact_pct')),
                "hint": f"Tension {fmt_pct(visual_metrics.get('tension_percentage'))}"
            })

        text_metrics = report_data.get('text_analytics', {})
        if text_metrics:
            insights.append({
                "id": "topic",
                "label": "Topic Coherence",
                "value": f"{text_metrics.get('topic_coherence_score', 0):.1f}/100",
                "hint": f"{text_metrics.get('keyword_coverage', {}).get('total_keywords', 0)} key terms"
            })

        # Flagged filler and pauses
        filler_events = (report_data.get('filler_analysis', {}) or {}).get('mumbling_instances', [])
        if filler_events:
            top_fillers = sorted(filler_events, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
            for item in top_fillers:
                timestamp = format_time(item.get('timestamp') or item.get('start'))
                label = item.get('word') or item.get('label') or 'filler'
                flags.append({
                    "type": "filler",
                    "label": label,
                    "timestamp": timestamp,
                    "detail": f"Detected filler '{label}' around {timestamp}"
                })

        pauses = (report_data.get('pauses_detailed') or [])[:4]
        for pause in pauses:
            if pause.get('duration', 0) >= 2:
                timestamp = format_time(pause.get('start'))
                flags.append({
                    "type": "pause",
                    "label": "Long Pause",
                    "timestamp": timestamp,
                    "detail": f"Pause lasted {pause.get('duration', 0):.1f}s"
                })

        improvements = report_data.get('improvements', [])[:3]
        for improvement in improvements:
            quick_prompts.append(f"How can I work on: {improvement}")

        quick_prompts.extend([
            "Summarize my top three strengths from this session.",
            "Design a 7-day drill plan to fix my weaknesses.",
            "Explain the biggest risk to my executive presence.",
            "Give me feedback on my opening 30 seconds."
        ])

        # Deduplicate quick prompts
        quick_prompts = list(dict.fromkeys(quick_prompts))[:8]

        return {
            "insights": insights,
            "flags": flags,
            "quick_prompts": quick_prompts
        }

# Global instance
chatbot_service = ChatbotService()







