"""
Transcript Enhancer Module
Uses AI to improve spoken transcript to formal, professional presentation style
"""
import logging
from typing import Dict, Optional
from ...services.openai_enhancer import openai_enhancer

logger = logging.getLogger(__name__)

class TranscriptEnhancer:
    """Enhances spoken transcript to formal presentation style"""
    
    def __init__(self):
        self.openai = openai_enhancer
    
    def enhance_transcript(self, transcript: str) -> Dict:
        """
        Enhance transcript to be more formal and professional
        
        Args:
            transcript: Raw spoken transcript
            
        Returns:
            Dictionary with enhanced transcript and improvement suggestions
        """
        try:
            if not transcript or not transcript.strip():
                return {
                    'original': transcript,
                    'enhanced': transcript,
                    'improvements': [],
                    'success': False
                }
            
            if not self.openai.enabled:
                logger.warning("OpenAI not available for transcript enhancement")
                return self._simple_enhancement(transcript)
            
            prompt = f"""You are an expert presentation coach. Improve this spoken transcript to make it more formal, professional, and suitable for a presentation or interview.

Original transcript:
{transcript}

Improve the transcript by:
1. Making it more formal and professional
2. Adding appropriate greetings and closings if missing
3. Improving sentence structure and clarity
4. Using more precise vocabulary
5. Adding transitions where needed
6. Maintaining the original meaning and key points

Respond in this JSON format:
{{
    "enhanced_transcript": "The improved, formal version of the transcript",
    "improvements": [
        {{
            "original": "original phrase or sentence",
            "improved": "improved version",
            "reason": "explanation of why this change was made"
        }}
    ],
    "key_changes": [
        "Brief summary of main improvements made"
    ]
}}

Example:
Original: "good morning I am here to talk about the topic"
Enhanced: "Good morning everyone. I am here today to present my views on this important topic."
Improvement: Added formality, greeting, and clearer structure.

Respond ONLY with valid JSON, no additional text."""

            response = self.openai.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert presentation coach. Always respond with valid JSON only, no markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            result = json.loads(result_text)
            
            return {
                'original': transcript,
                'enhanced': result.get('enhanced_transcript', transcript),
                'improvements': result.get('improvements', []),
                'key_changes': result.get('key_changes', []),
                'success': True
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in transcript enhancement: {e}")
            return self._simple_enhancement(transcript)
        except Exception as e:
            logger.error(f"Error enhancing transcript: {e}")
            return self._simple_enhancement(transcript)
    
    def _simple_enhancement(self, transcript: str) -> Dict:
        """Simple rule-based enhancement as fallback"""
        try:
            enhanced = transcript.strip()
            
            # Capitalize first letter
            if enhanced:
                enhanced = enhanced[0].upper() + enhanced[1:]
            
            # Add period if missing
            if enhanced and not enhanced.endswith(('.', '!', '?')):
                enhanced += '.'
            
            improvements = []
            
            # Simple replacements
            replacements = {
                'good morning': 'Good morning everyone',
                'i am here to talk': 'I am here to present',
                'talk about': 'discuss',
                'thing': 'matter',
                'stuff': 'material',
            }
            
            for old, new in replacements.items():
                if old.lower() in enhanced.lower():
                    improvements.append({
                        'original': old,
                        'improved': new,
                        'reason': 'More formal and professional phrasing'
                    })
                    enhanced = enhanced.replace(old, new)
            
            return {
                'original': transcript,
                'enhanced': enhanced,
                'improvements': improvements,
                'key_changes': ['Basic formatting and capitalization applied'] if improvements else [],
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in simple enhancement: {e}")
            return {
                'original': transcript,
                'enhanced': transcript,
                'improvements': [],
                'key_changes': [],
                'success': False
            }

# Global instance
transcript_enhancer = TranscriptEnhancer()















