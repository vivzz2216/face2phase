"""
Optional OpenAI API integration for enhanced analysis
"""
import os
import logging
from typing import Dict, Optional

try:
    from openai import OpenAI
    OPENAI_PACKAGE_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_PACKAGE_AVAILABLE = False

from ..core.settings import OPENAI_API_KEY, USE_OPENAI_API

logger = logging.getLogger(__name__)

class OpenAIEnhancer:
    """Optional OpenAI API integration for enhanced analysis"""
    
    def __init__(self):
        self.client = None
        self.enabled = False
        
        if not OPENAI_PACKAGE_AVAILABLE:
            logger.warning("OpenAI package not installed. Install with: pip install openai")
            return
        
        if USE_OPENAI_API and OPENAI_API_KEY:
            try:
                self.client = OpenAI(api_key=OPENAI_API_KEY)
                self.enabled = True
                logger.info("OpenAI API integration enabled")
            except Exception as e:
                logger.warning(f"OpenAI API initialization failed: {e}")
                self.enabled = False
    
    def enhance_summary(self, transcript: str, basic_summary: str) -> str:
        """
        Generate enhanced summary using GPT-4
        
        Args:
            transcript: Full transcript
            basic_summary: Basic summary from local analysis
            
        Returns:
            Enhanced summary or basic summary if API unavailable
        """
        if not self.enabled or not transcript.strip():
            return basic_summary
        
        try:
            prompt = f"""
            Analyze this speech transcript and provide a concise, actionable summary focusing on:
            1. Key messages and main points
            2. Communication strengths
            3. Areas for improvement
            4. Specific recommendations
            
            Transcript: {transcript[:2000]}  # Limit to avoid token limits
            
            Keep the summary under 200 words and make it practical for communication improvement.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective model
                messages=[
                    {"role": "system", "content": "You are a communication coach analyzing speech transcripts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            enhanced_summary = response.choices[0].message.content.strip()
            logger.info("Enhanced summary generated successfully")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return basic_summary
    
    def generate_coaching_tips(self, analysis_results: Dict) -> list:
        """
        Generate personalized coaching tips using GPT-4
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            List of personalized coaching tips
        """
        if not self.enabled:
            return []
        
        try:
            # Extract key metrics
            voice_score = analysis_results.get('voice_confidence', 0)
            facial_score = analysis_results.get('facial_confidence', 0)
            vocab_score = analysis_results.get('vocabulary_score', 0)
            filler_count = analysis_results.get('filler_word_count', 0)
            speaking_rate = analysis_results.get('speaking_rate_wpm', 0)
            
            prompt = f"""
            Based on this communication analysis, provide 3-5 specific, actionable coaching tips:
            
            Voice Confidence: {voice_score}/100
            Facial Confidence: {facial_score}/100
            Vocabulary Score: {vocab_score}/100
            Filler Words: {filler_count}
            Speaking Rate: {speaking_rate} WPM
            
            Focus on the lowest scoring areas and provide practical exercises or techniques.
            Keep each tip under 50 words and make them specific and actionable.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert communication coach providing personalized feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            tips_text = response.choices[0].message.content.strip()
            # Parse tips into list
            tips = [tip.strip() for tip in tips_text.split('\n') if tip.strip() and tip.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-', '•'))]
            
            logger.info(f"Generated {len(tips)} coaching tips")
            return tips
            
        except Exception as e:
            logger.error(f"OpenAI coaching tips error: {e}")
            return []
    
    def analyze_sentiment_tone(self, transcript: str) -> Dict:
        """
        Analyze sentiment and tone using GPT-4
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Dictionary with sentiment analysis
        """
        if not self.enabled or not transcript.strip():
            return {"sentiment": "neutral", "confidence": 0.5, "tone": "professional"}
        
        try:
            prompt = f"""
            Analyze the sentiment and tone of this speech transcript:
            
            {transcript[:1500]}  # Limit for token efficiency
            
            Provide:
            1. Overall sentiment (positive, negative, neutral)
            2. Confidence level (0-1)
            3. Tone description (professional, casual, enthusiastic, etc.)
            4. Emotional undertones
            
            Respond in JSON format.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing speech sentiment and tone. Respond only in valid JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.choices[0].message.content.strip())
            logger.info("Sentiment analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI sentiment analysis error: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "tone": "professional"}

    def generate_advanced_feedback(self, analysis_data: Dict) -> Dict:
        """
        Generate comprehensive feedback using OpenAI
        
        Args:
            analysis_data: Combined analysis results
            
        Returns:
            Enhanced feedback with AI-generated insights
        """
        if not self.enabled:
            return {"enhanced_feedback": False, "reason": "OpenAI not available"}
        
        try:
            # Prepare context for OpenAI
            context = self._prepare_analysis_context(analysis_data)
            
            # Generate comprehensive feedback
            prompt = f"""
            As an expert communication coach, analyze this communication data and provide detailed feedback:
            
            {context}
            
            Please provide:
            1. Overall communication score (0-100)
            2. Strengths and areas for improvement
            3. Specific actionable recommendations
            4. Personalized coaching tips
            5. Progress tracking suggestions
            
            Format as JSON with detailed explanations.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert communication coach providing detailed, actionable feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            feedback_text = response.choices[0].message.content
            
            return {
                "enhanced_feedback": True,
                "ai_coaching": feedback_text,
                "model_used": "gpt-4o-mini"
            }
            
        except Exception as e:
            logger.error(f"OpenAI feedback generation failed: {e}")
            return {"enhanced_feedback": False, "error": str(e)}
    
    def generate_personalized_coaching(self, user_history: list) -> Dict:
        """
        Generate personalized coaching based on user history
        
        Args:
            user_history: List of previous analysis results
            
        Returns:
            Personalized coaching recommendations
        """
        if not self.enabled:
            return {"coaching": False, "reason": "OpenAI not available"}
        
        try:
            # Analyze user progress
            progress_summary = self._analyze_user_progress(user_history)
            
            prompt = f"""
            Based on this user's communication analysis history, provide personalized coaching:
            
            {progress_summary}
            
            Generate:
            1. Progress assessment
            2. Personalized improvement plan
            3. Specific exercises and practices
            4. Goal setting recommendations
            5. Motivation and encouragement
            
            Be specific and actionable.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a personalized communication coach."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.6
            )
            
            return {
                "coaching": True,
                "personalized_plan": response.choices[0].message.content,
                "model_used": "gpt-4o-mini"
            }
            
        except Exception as e:
            logger.error(f"Personalized coaching generation failed: {e}")
            return {"coaching": False, "error": str(e)}
    
    def _prepare_analysis_context(self, analysis_data: Dict) -> str:
        """Prepare analysis context for OpenAI"""
        context_parts = []
        
        # Audio analysis
        if 'audio_analysis' in analysis_data:
            audio = analysis_data['audio_analysis']
            context_parts.append(f"Voice Analysis: Confidence {audio.get('voice_confidence_score', 0)}/100, "
                               f"Clarity {audio.get('speech_clarity_score', 0)}/100")
        
        # Facial analysis
        if 'facial_analysis' in analysis_data:
            facial = analysis_data['facial_analysis']
            context_parts.append(f"Facial Analysis: Confidence {facial.get('facial_confidence_score', 0)}/100, "
                               f"Engagement {facial.get('engagement_score', 0)}/100")
        
        # Text analysis
        if 'text_analysis' in analysis_data:
            text = analysis_data['text_analysis']
            context_parts.append(f"Text Analysis: Vocabulary {text.get('vocabulary_score', 0)}/100, "
                               f"Clarity {text.get('text_clarity_score', 0)}/100")
        
        return "\n".join(context_parts)
    
    def _analyze_user_progress(self, user_history: list) -> str:
        """Analyze user progress over time"""
        if not user_history:
            return "No previous analysis history available."
        
        # Extract scores over time
        scores = []
        for analysis in user_history:
            overall_score = analysis.get('overall_score', 0)
            scores.append(overall_score)
        
        # Calculate trends
        if len(scores) >= 2:
            trend = "improving" if scores[-1] > scores[0] else "declining"
            avg_score = sum(scores) / len(scores)
            return f"User has {len(scores)} analyses. Average score: {avg_score:.1f}. Trend: {trend}."
        
        return f"User has {len(scores)} analysis. Latest score: {scores[0]}."

# Global OpenAI enhancer instance
openai_enhancer = OpenAIEnhancer()
