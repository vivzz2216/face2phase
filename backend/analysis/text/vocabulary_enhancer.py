"""
Vocabulary Enhancer Module
Suggests formal alternatives for basic/informal words
"""
import logging
import re
from typing import Dict, List, Optional
try:
    from ...services.openai_enhancer import openai_enhancer
except ImportError:
    from services.openai_enhancer import openai_enhancer

logger = logging.getLogger(__name__)

# Common phrases and greetings that should NOT be replaced
PROTECTED_PHRASES = [
    'good morning', 'good afternoon', 'good evening', 'good night',
    'good day', 'good luck', 'good bye', 'goodbye',
    'thank you', 'thanks', 'please', 'excuse me',
    'how are you', 'nice to meet you', 'pleased to meet you',
    'i am', 'i\'m', 'my name is', 'i come from',
    'from', 'to', 'the', 'a', 'an', 'and', 'or', 'but',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'may', 'might',
    'this', 'that', 'these', 'those', 'here', 'there',
    'in', 'on', 'at', 'by', 'for', 'with', 'about', 'into',
    'of', 'off', 'over', 'under', 'up', 'down', 'out'
]

# Common basic-to-formal word mappings (only for words that are actually weak)
BASIC_TO_FORMAL = {
    'good': ['excellent', 'outstanding', 'superior', 'remarkable'],
    'bad': ['poor', 'inadequate', 'substandard', 'unsatisfactory'],
    'big': ['substantial', 'significant', 'considerable', 'extensive'],
    'small': ['modest', 'minimal', 'limited', 'compact'],
    'talk': ['present', 'discuss', 'address', 'communicate'],
    'say': ['state', 'express', 'articulate', 'convey'],
    'think': ['consider', 'contemplate', 'reflect', 'analyze'],
    'know': ['understand', 'comprehend', 'recognize', 'acknowledge'],
    'get': ['obtain', 'acquire', 'receive', 'attain'],
    'give': ['provide', 'deliver', 'offer', 'present'],
    'make': ['create', 'develop', 'produce', 'construct'],
    'do': ['perform', 'execute', 'accomplish', 'implement'],
    'thing': ['matter', 'aspect', 'element', 'component'],
    'stuff': ['material', 'content', 'substance', 'information'],
    'lots of': ['numerous', 'multiple', 'several', 'various'],
    'many': ['numerous', 'multiple', 'several', 'various'],
    'very': ['extremely', 'highly', 'particularly', 'significantly'],
    'really': ['truly', 'genuinely', 'particularly', 'significantly'],
    'just': ['simply', 'merely', 'only', 'solely'],
    'like': ['such as', 'for example', 'including', 'similar to'],
    'try': ['attempt', 'endeavor', 'strive', 'seek'],
    'want': ['desire', 'wish', 'require', 'need'],
    'use': ['utilize', 'employ', 'apply', 'implement'],
    'help': ['assist', 'support', 'facilitate', 'aid'],
    'start': ['begin', 'commence', 'initiate', 'launch'],
    'end': ['conclude', 'complete', 'finish', 'terminate'],
    'show': ['demonstrate', 'illustrate', 'display', 'present'],
    'find': ['discover', 'identify', 'locate', 'determine'],
    'look': ['examine', 'review', 'analyze', 'inspect'],
    'change': ['modify', 'alter', 'adjust', 'transform'],
}

class VocabularyEnhancer:
    """Suggests formal alternatives for basic words"""
    
    def __init__(self):
        self.openai = openai_enhancer
    
    def enhance_vocabulary(self, transcript: str) -> Dict:
        """
        Detect basic words and suggest formal alternatives
        
        Args:
            transcript: Input transcript
            
        Returns:
            Dictionary with vocabulary enhancements
        """
        try:
            if not transcript or not transcript.strip():
                return {
                    'enhancements': [],
                    'total_suggestions': 0,
                    'success': False
                }
            
            # Find basic words in transcript
            basic_words_found = self._detect_basic_words(transcript)
            
            if not basic_words_found:
                return {
                    'enhancements': [],
                    'total_suggestions': 0,
                    'success': True,
                    'message': 'Your vocabulary is already quite formal.'
                }
            
            # Generate suggestions
            enhancements = []
            for word_data in basic_words_found:
                word = word_data['word']
                context = word_data['context']
                formal_options = self._get_formal_alternatives(word, context)
                
                if formal_options:
                    enhancements.append({
                        'word': word,
                        'context': context,
                        'suggestions': formal_options,
                        'reason': self._get_replacement_reason(word, formal_options[0])
                    })
            
            # Limit to most important suggestions
            enhancements = enhancements[:10]
            
            return {
                'enhancements': enhancements,
                'total_suggestions': len(enhancements),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error enhancing vocabulary: {e}")
            return {
                'enhancements': [],
                'total_suggestions': 0,
                'success': False
            }
    
    def _detect_basic_words(self, transcript: str) -> List[Dict]:
        """Detect basic words in transcript with context, excluding greetings and common phrases"""
        basic_words_found = []
        words = re.findall(r'\b\w+\b', transcript.lower())
        sentences = re.split(r'[.!?]+', transcript)
        
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            sentence_original = sentence.strip()
            
            # Skip if sentence is empty
            if not sentence_lower:
                continue
            
            # Check if sentence contains protected phrases (greetings, etc.)
            is_protected = False
            for phrase in PROTECTED_PHRASES:
                if phrase in sentence_lower:
                    is_protected = True
                    break
            
            # If it's a protected phrase, skip this sentence entirely
            if is_protected:
                continue
            
            for basic_word, formal_options in BASIC_TO_FORMAL.items():
                # Check for exact word match
                word_pattern = re.compile(r'\b' + re.escape(basic_word) + r'\b', re.IGNORECASE)
                matches = word_pattern.finditer(sentence_original)
                
                for match in matches:
                    word = match.group()
                    word_lower = word.lower()
                    
                    # Skip if word is part of a protected phrase
                    # Check surrounding words (2 words before and after)
                    match_start = match.start()
                    match_end = match.end()
                    context_start = max(0, match_start - 20)
                    context_end = min(len(sentence_original), match_end + 20)
                    surrounding_text = sentence_original[context_start:context_end].lower()
                    
                    is_in_protected_phrase = False
                    for phrase in PROTECTED_PHRASES:
                        if phrase in surrounding_text:
                            is_in_protected_phrase = True
                            break
                    
                    if is_in_protected_phrase:
                        continue
                    
                    # Special handling for "good" - only suggest replacement if it's not in a greeting
                    if word_lower == 'good':
                        # Check if it's followed by time-of-day words
                        next_words = sentence_lower[match_end:match_end+15].split()[:2]
                        if any(time_word in next_words for time_word in ['morning', 'afternoon', 'evening', 'night', 'day']):
                            continue
                        # Check if it's "good luck", "good bye", etc.
                        if any(phrase in sentence_lower[max(0, match_start-5):match_end+10] for phrase in ['good luck', 'good bye', 'goodbye']):
                            continue
                    
                    # Avoid duplicates
                    if not any(w['word'].lower() == word.lower() and w['context'] == sentence_original for w in basic_words_found):
                        basic_words_found.append({
                            'word': word,
                            'context': sentence_original
                        })
        
        return basic_words_found
    
    def _get_formal_alternatives(self, word: str, context: str) -> List[str]:
        """Get formal alternatives for a word"""
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        
        # Check dictionary first
        if word_lower in BASIC_TO_FORMAL:
            return BASIC_TO_FORMAL[word_lower]
        
        # Try OpenAI for context-aware suggestions
        if self.openai.enabled:
            try:
                return self._get_ai_suggestions(word, context)
            except Exception as e:
                logger.error(f"Error getting AI suggestions: {e}")
        
        return []
    
    def _get_ai_suggestions(self, word: str, context: str) -> List[str]:
        """Get AI-powered formal alternatives"""
        prompt = f"""Provide 3-4 formal, professional alternatives for the word "{word}" in this context:
"{context}"

Respond with only a JSON array of alternatives, no explanations.
Example: ["alternative1", "alternative2", "alternative3"]"""

        try:
            response = self.openai.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a vocabulary expert. Respond with only a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.5
            )
            
            import json
            suggestions_text = response.choices[0].message.content.strip()
            if suggestions_text.startswith("```"):
                suggestions_text = suggestions_text.split("```")[1]
                if suggestions_text.startswith("json"):
                    suggestions_text = suggestions_text[4:]
            
            suggestions = json.loads(suggestions_text)
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            logger.error(f"Error parsing AI suggestions: {e}")
            return []
    
    def _get_replacement_reason(self, word: str, replacement: str) -> str:
        """Get reason for word replacement"""
        reasons = {
            'good': 'More precise and impactful',
            'bad': 'More professional and specific',
            'big': 'More formal and descriptive',
            'small': 'More precise measurement',
            'talk': 'More appropriate for presentations',
            'say': 'More formal communication verb',
            'think': 'More analytical and thoughtful',
            'know': 'More comprehensive understanding',
            'get': 'More formal acquisition verb',
            'give': 'More professional delivery verb',
            'make': 'More creative and intentional',
            'do': 'More action-oriented',
            'thing': 'More specific and clear',
            'stuff': 'More formal and organized',
            'very': 'More impactful intensifier',
            'really': 'More professional emphasis',
        }
        
        return reasons.get(word.lower(), 'More formal and professional alternative')

# Global instance
vocabulary_enhancer = VocabularyEnhancer()



