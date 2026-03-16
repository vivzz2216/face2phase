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
        """
        Check if user question is related to the uploaded video/audio content
        Blocks general knowledge queries (sports, celebrities, politics, etc.)
        """
        message_lower = user_message.lower().strip()
        
        # General knowledge topics that should be BLOCKED
        blocked_keywords = {
            'messi', 'ronaldo', 'football', 'soccer', 'cricket', 'basketball',
            'sports', 'celebrity', 'actor', 'actress', 'singer', 'president',
            'minister', 'politics', 'capital', 'population', 'movie', 'film'
        }
        
        # Check for blocked keywords
        message_words = set(message_lower.split())
        if message_words & blocked_keywords:
            logger.info(f"[CHATBOT] Blocked general knowledge query: {message_words & blocked_keywords}")
            return False
        
        # Presentation-related keywords (ALLOWED)
        allowed_keywords = {
            'score', 'voice', 'speaking', 'facial', 'eye contact', 'transcript',
            'improve', 'feedback', 'my', 'i', 'presentation', 'video', 'analysis'
        }
        
        # If question contains allowed keywords, it's relevant
        if message_words & allowed_keywords:
            return True
        
        # Default: allow (to avoid false positives)
        return True
