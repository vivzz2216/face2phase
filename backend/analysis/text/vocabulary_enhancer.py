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
    'my name is', 'i come from',
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

# Phrase-level enhancements (more powerful - these match multi-word patterns)
# Format: (pattern_regex, replacement_function, description)
PHRASE_PATTERNS = [
    # USER-REQUESTED SPECIFIC PATTERNS (HIGH PRIORITY)
    (r'\ba topic which is\b',
     'a topic called',
     '"a topic which is" → "a topic called"'),
    
    (r'\bthese all\b',
     'all these',
     '"these all" → "all these"'),
    
    (r'\bvery old\b',
     'ancient',
     '"very old" → "ancient"'),
    
    (r'\bvery traditional\b',
     'highly traditional',
     '"very traditional" → "highly traditional"'),
    
    (r'\bdance form\b',
     'dance style',
     '"dance form" → "dance style"'),
    
    (r'\boriginated\b',
     'started',
     '"originated" → "started"'),
    
    (r'\bbasically\b',
     'mainly',
     '"basically" → "mainly"'),
    
    # Skill/Competency Phrases
    (r'\b(I am |I\'m |I was |I\'ve been )?very good at\b', 
     lambda m: f"{m.group(1) or ''}proficient in" if m.group(1) else "proficient in",
     '"very good at" → "proficient in"'),
    
    (r'\b(I am |I\'m |I was )good at\b',
     lambda m: "I excel at" if m.group(1).strip() in ["I am", "I'm"] else "I excelled at",
     '"I am good at" → "I excel at"'),

    
    (r'\b(I am |I\'m )really good at\b',
     lambda m: "I am highly skilled in",
     '"I am really good at" → "I am highly skilled in"'),
    
    (r'\b(I have |I\'ve )good( knowledge| experience)? (of|in|with)\b',
     lambda m: f"I possess strong{m.group(2) or ''} {m.group(3)}",
     '"I have good knowledge of" → "I possess strong knowledge of"'),
    
    # Intensity/Degree Modifiers
    (r'\bvery (important|essential|critical|vital)\b',
     lambda m: f"crucial" if m.group(1) in ['important', 'essential'] else f"absolutely {m.group(1)}",
     '"very important" → "crucial"'),
    
    (r'\bvery (big|large|huge)\b',
     lambda m: "substantial" if m.group(1) == 'big' else "extensive",
     '"very big" → "substantial"'),
    
    (r'\bvery (small|little|tiny)\b',
     lambda m: "minimal",
     '"very small" → "minimal"'),
    
    (r'\bvery (happy|pleased|satisfied)\b',
     lambda m: "delighted" if m.group(1) == 'happy' else "highly satisfied",
     '"very happy" → "delighted"'),
    
    (r'\breally (want|need|like)\b',
     lambda m: f"strongly {'desire' if m.group(1) == 'want' else 'require' if m.group(1) == 'need' else 'appreciate'}",
     '"really want" → "strongly desire"'),
    
    # Demonstrative Phrases  
    (r'\b(I think|I believe) that\b',
     lambda m: "I contend that" if 'think' in m.group(1) else "I am convinced that",
     '"I think that" → "I contend that"'),
    
    (r'\bI would like to\b',
     'I wish to',
     '"I would like to" → "I wish to"'),
    
    (r'\bI want to\b',
     'I aim to',
     '"I want to" → "I aim to"'),
    
    # Capability Phrases
    (r'\b(I can|I could) (easily |quickly )?(do|make|create|handle)\b',
     lambda m: f"I am capable of {m.group(2) or ''}{m.group(3).rstrip('e') + 'ing'}",
     '"I can do" → "I am capable of doing"'),
    
    (r'\b(I am able to|I\'m able to)\b',
     'I am capable of',
     '"I am able to" → "I am capable of"'),
    
    # Experience/Background
    (r'\b(I have been|I\'ve been) (working|doing|studying)\b',
     lambda m: f"I have {'worked' if m.group(2) == 'working' else 'engaged' if m.group(2) == 'doing' else 'studied'}",
     '"I have been working" → "I have worked"'),
    
    (r'\b(I have|I\'ve) done (a lot of|lots of|many)\b',
     lambda m: "I have completed numerous",
     '"I have done a lot of" → "I have completed numerous"'),
    
    # Interest/Passion
    (r'\b(I really|I) like (to )?\b',
     lambda m: f"I am passionate about {m.group(2) or ''}",
     '"I really like" → "I am passionate about"'),
    
    (r'\b(I am |I\'m )interested in\b',
     lambda m: "I have a keen interest in",
     '"I am interested in" → "I have a keen interest in"'),
    
    # General Improvements
    (r'\ba lot of\b',
     'numerous',
     '"a lot of" → "numerous"'),
    
    (r'\blots of\b',
     'many',
     '"lots of" → "many"'),
    
    (r'\bkind of\b',
     'somewhat',
     '"kind of" → "somewhat"'),
    
    (r'\bsort of\b',
     'rather',
     '"sort of" → "rather"'),
    
    #  Demonstrative improvements
    (r'\bthis is (a |an )?(good|great|nice)\b',
     lambda m: f"this represents {'an' if m.group(1) == 'an ' else 'a'} {'excellent' if m.group(2) == 'good' else 'outstanding' if m.group(2) == 'great' else 'pleasant'}",
     '"this is good" → "this represents an excellent"'),
]


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
            
            logger.info(f"Vocabulary enhancer detected {len(basic_words_found)} words/phrases")
            for item in basic_words_found[:5]:  # Log first 5
                logger.info(f"  - '{item['word']}' (is_phrase: {item.get('is_phrase', False)})")
            
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
                
                # Check if this is a phrase with a pre-computed suggestion
                if word_data.get('is_phrase') and word_data.get('suggestion'):
                    # For phrases, use the pre-computed suggestion
                    formal_options = [word_data['suggestion']]
                else:
                    # For individual words, use the dictionary
                    formal_options = self._get_formal_alternatives(word, context)
                
                if formal_options:
                    enhancements.append({
                        'word': word,
                        'context': context,
                        'suggestions': formal_options if isinstance(formal_options, list) else [formal_options],
                        'reason': self._get_replacement_reason(word, formal_options[0] if isinstance(formal_options, list) else formal_options)
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
            
            # PRIORITY 1: Check phrase patterns FIRST (more powerful)
            for pattern, replacement, description in PHRASE_PATTERNS:
                pattern_regex = re.compile(pattern, re.IGNORECASE)
                matches = pattern_regex.finditer(sentence_original)
                
                for match in matches:
                    matched_phrase = match.group()
                    
                    # Calculate replacement
                    if callable(replacement):
                        try:
                            suggested_replacement = replacement(match)
                        except Exception as e:
                            logger.error(f"Error in replacement function: {e}")
                            continue
                    else:
                        suggested_replacement = replacement
                    
                    # Avoid duplicates
                    if not any(w['word'].lower() == matched_phrase.lower() and w['context'] == sentence_original for w in basic_words_found):
                        basic_words_found.append({
                            'word': matched_phrase,
                            'context': sentence_original,
                            'suggestion': suggested_replacement,
                            'is_phrase': True
                        })
            
            # PRIORITY 2: Check individual words (only if not already matched by phrases)
            for basic_word, formal_options in BASIC_TO_FORMAL.items():
                # Check for exact word match
                word_pattern = re.compile(r'\b' + re.escape(basic_word) + r'\b', re.IGNORECASE)
                matches = word_pattern.finditer(sentence_original)
                
                for match in matches:
                    word = match.group()
                    word_lower = word.lower()
                    
                    # Skip if this word is part of a phrase we already matched
                    already_matched_in_phrase = False
                    for found_item in basic_words_found:
                        if found_item.get('is_phrase') and word_lower in found_item['word'].lower():
                            already_matched_in_phrase = True
                            break
                    
                    if already_matched_in_phrase:
                        continue
                    
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
                            'context': sentence_original,
                            'is_phrase': False
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



