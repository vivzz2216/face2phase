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
        """Simple rule-based enhancement as fallback (Sentence-level)"""
        try:
            import re
            
            # Phrase replacement rules
            phrase_replacements = [
                # USER-REQUESTED SPECIFIC PATTERNS (HIGH PRIORITY)
                (r'\ba topic which is\b', 'a topic called', '"a topic which is" → "a topic called"'),
                (r'\bthese all\b', 'all these', '"these all" → "all these"'),
                (r'\bvery old\b', 'ancient', '"very old" → "ancient"'),
                (r'\bvery traditional\b', 'highly traditional', '"very traditional" → "highly traditional"'),
                (r'\bdance form\b', 'dance style', '"dance form" → "dance style"'),
                (r'\boriginated\b', 'started', '"originated" → "started"'),
                (r'\bbasically\b', 'mainly', '"basically" → "mainly"'),
                
                # General patterns
                (r'\b(I am |I\'m )very good at\b', r'\1proficient in', '"very good at" → "proficient in"'),
                (r'\b(I am |I\'m )good at\b', r'I excel at', '"I am good at" → "I excel at"'),
                (r'\ba lot of\b', 'numerous', '"a lot of" → "numerous"'),
                (r'\blots of\b', 'many', '"lots of" → "many"'),
                (r'\bvery (big|large)\b', 'substantial', '"very big" → "substantial"'),
                (r'\bvery (small|little)\b', 'minimal', '"very small" → "minimal"'),
                (r'\bvery (important|good|old)\b', lambda m: 'crucial' if m.group(1) == 'important' else 'excellent' if m.group(1) == 'good' else 'ancient', '"very important/good/old" → enhanced'),
                (r'\breally (want|like)\b', lambda m: f"strongly {'desire' if m.group(1) == 'want' else 'appreciate'}", '"really want" → "strongly desire"'),
                (r'\bI want to\b', 'I aim to', '"I want to" → "I aim to"'),
                (r'\bI would like to\b', 'I wish to', '"I would like to" → "I wish to"'),
                (r'\bkind of\b', 'somewhat', '"kind of" → "somewhat"'),
                (r'\bsort of\b', 'rather', '"sort of" → "rather"'),
                (r'\btalk about\b', 'discuss', '"talk about" → "discuss"'),
                (r'\bwe\'ll talk about\b', 'we will discuss', '"we\'ll talk about" → "we will discuss"'),
            ]

            word_replacements = {
                r'\bthing\b': 'matter',
                r'\bstuff\b': 'material',
                r'\bget\b': 'obtain',
                r'\bgive\b': 'provide',
            }

            # Split logic: look for punctuation followed by space
            # If no punctuation, treat as single block
            raw_sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
            if not raw_sentences: 
                raw_sentences = [transcript.strip()]

            enhanced_sentences = []
            improvements = []
            
            filler_pattern = r'\b(uh|um|ah|er|like|you know|I mean|basically|actually)\b[,\s]*'

            for sent in raw_sentences:
                if not sent.strip(): continue
                
                original_sent = sent
                working_sent = sent
                reasons = []

                # 1. Filler removal (local to sentence)
                filler_matches = re.findall(filler_pattern, working_sent, re.IGNORECASE)
                if filler_matches:
                    working_sent = re.sub(filler_pattern, ' ', working_sent, flags=re.IGNORECASE)
                    reasons.append("Removed filler words")

                # 2. Cleanup spaces (first pass)
                working_sent = re.sub(r'\s+', ' ', working_sent).strip()

                # 3. Phrase replacements
                for pattern, replacement, desc in phrase_replacements:
                    if re.search(pattern, working_sent, re.IGNORECASE):
                        before_sub = working_sent
                        if callable(replacement):
                            working_sent = re.sub(pattern, lambda m: replacement(m), working_sent, flags=re.IGNORECASE)
                        else:
                            working_sent = re.sub(pattern, replacement, working_sent, flags=re.IGNORECASE)
                        
                        if working_sent != before_sub:
                            reasons.append(desc.split('→')[1].strip().replace('"', '') if '→' in desc else desc)

                # 4. Word replacements
                for pattern, new_word in word_replacements.items():
                    if re.search(pattern, working_sent, re.IGNORECASE):
                        before_sub = working_sent
                        working_sent = re.sub(pattern, new_word, working_sent, flags=re.IGNORECASE)
                        if working_sent != before_sub:
                            clean_pattern = pattern.replace(r'\b', '').replace(r'\\', '')
                            reasons.append(f"Used '{new_word}' instead of {clean_pattern}")

                # 5. Grammar/Capitalization
                # Capitalize first letter
                if working_sent:
                    working_sent = working_sent[0].upper() + working_sent[1:]
                
                # Fix spacing around punctuation
                working_sent = re.sub(r'\s+([,.])', r'\1', working_sent)
                working_sent = re.sub(r',\s*,', ',', working_sent)
                working_sent = re.sub(r'\.\s*\.', '.', working_sent)
                
                # Add period if missing (only if it looks like a complete sentence)
                if len(working_sent) > 5 and not working_sent.endswith(('.', '!', '?')):
                    working_sent += '.'

                enhanced_sentences.append(working_sent)

                # Record improvement if changed significantly
                # (Ignore simple period addition to avoid clutter if that's the only change)
                if working_sent != original_sent:
                    # Filter out changes that are just space/period fixes unless that was the goal
                    clean_orig = re.sub(r'\s+', ' ', original_sent).strip()
                    clean_new = re.sub(r'\s+', ' ', working_sent).strip()
                    
                    if clean_orig.lower() != clean_new.lower() or reasons:
                        reason_str = "; ".join(list(set(reasons)))
                        if not reason_str: reason_str = "Improved grammar and formatting"
                        
                        improvements.append({
                            'original': original_sent,
                            'enhanced': working_sent,
                            'reason': reason_str
                        })

            final_text = " ".join(enhanced_sentences)
            
            return {
                'original': transcript,
                'enhanced': final_text,
                'improvements': improvements[:8], # Limit to top 8
                'key_changes': list(set([imp['reason'] for imp in improvements]))[:5],
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















