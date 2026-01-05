"""
Enhanced Word Analysis Module
Detects weak words, filler words, vocabulary richness, and provides improvement suggestions
"""
import logging
import re
from typing import Dict, List, Tuple
from collections import Counter
from ...core.settings import FILLER_WORDS

# Try to import spacy for context-aware analysis
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False

# Try to import ML pattern recognizer
try:
    from ...services.ml_pattern_recognizer import ml_pattern_recognizer
    ML_PATTERN_AVAILABLE = True
except ImportError:
    ml_pattern_recognizer = None
    ML_PATTERN_AVAILABLE = False

logger = logging.getLogger(__name__)

class WordAnalyzer:
    """Analyzes words for weak language, fillers, and vocabulary quality"""
    
    def __init__(self):
        """Initialize WordAnalyzer with NLP model if available"""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                import spacy
                # Try to load model, fallback to basic if not available
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("Loaded spaCy model for context-aware analysis")
                except OSError:
                    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_md")
                    self.nlp = None
            except Exception as e:
                logger.warning(f"Could not load spaCy: {e}")
                self.nlp = None
    
    # Comprehensive weak words list
    WEAK_WORDS = {
        # Uncertainty words
        "just": "Try using more specific time references or removing entirely",
        "really": "Use stronger, more precise adjectives",
        "very": "Replace with more descriptive words (e.g., 'very good' → 'excellent')",
        "quite": "Be more specific about the degree",
        "pretty": "Use 'quite' or 'rather' for formal contexts, or be more specific",
        "sort of": "Use 'somewhat' or 'to some extent' for clarity",
        "kind of": "Use 'somewhat' or 'to some extent' for clarity",
        "maybe": "Use 'possibly', 'perhaps', or state confidence level",
        "perhaps": "Acceptable, but consider being more direct",
        "probably": "State probability more precisely (e.g., 'likely', 'most likely')",
        "might": "Use 'may' for formal contexts or be more specific",
        "could": "Use 'can' if expressing ability, 'may' if expressing possibility",
        
        # Filler-like words used as weak words
        "like": "Remove when used as filler, use 'such as' for examples",
        "you know": "Remove - it's a filler phrase",
        "well": "Remove when used as filler at sentence start",
        "actually": "Only use when correcting a misconception",
        "basically": "Remove - it undermines your message",
        "literally": "Only use when something actually happened literally",
        
        # Vague qualifiers
        "thing": "Use specific nouns (e.g., 'thing' → 'concept', 'element', 'component')",
        "stuff": "Use specific nouns",
        "things": "Use specific nouns",
        "something": "Be more specific about what you're referring to",
        "anything": "Be more specific",
        "nothing": "Use 'no evidence' or 'no data' instead",
        "everything": "Be more specific about what you mean",
        
        # Weak verbs
        "got": "Use 'have', 'received', 'obtained' instead",
        "get": "Use 'obtain', 'receive', 'acquire' instead",
        "gonna": "Use 'going to' or 'will'",
        "wanna": "Use 'want to'",
        "gotta": "Use 'have to' or 'must'",
        
        # Weak intensifiers
        "totally": "Remove or use 'completely', 'entirely'",
        "completely": "Acceptable, but consider if 'fully' or 'entirely' is better",
        "totally": "Remove or use more specific language",
    }
    
    # Extended filler words including mumbling patterns
    EXTENDED_FILLERS = [
        "um", "uh", "ah", "eh", "er", "hmm", "erm", "umm", "uhh", "ahh", "ehh", "err",
        "um-um", "uh-uh", "ah-ah", "er-er",
        "like", "you know", "well", "actually", "basically", "literally",
        "kinda", "sorta", "gonna", "wanna", "gotta",
        # Mumbling patterns
        "mhm", "uh-huh", "mm-hmm", "yeah", "yep", "yup"
    ]
    
    # Vocabulary improvement suggestions
    VOCABULARY_SUGGESTIONS = {
        "low_richness": "Your vocabulary has limited variety. Try reading more diverse content and using a thesaurus.",
        "high_repetition": "You're repeating words frequently. Consider using synonyms and varied sentence structures.",
        "simple_words": "Use more sophisticated vocabulary appropriate for professional contexts.",
        "weak_verbs": "Replace weak verbs with stronger, more specific action words."
    }
    
    def analyze_weak_words(self, transcript: str) -> Dict:
        """
        Analyze weak words in transcript
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Dictionary with weak words analysis
        """
        try:
            if not transcript or not transcript.strip():
                return {
                    "weak_words_found": [],
                    "weak_word_count": 0,
                    "weak_word_percentage": 0.0,
                    "weak_word_breakdown": {},
                    "total_words": 0
                }
            
            # Clean and tokenize
            words = self._tokenize(transcript)
            total_words = len(words)
            
            if total_words == 0:
                return {
                    "weak_words_found": [],
                    "weak_word_count": 0,
                    "weak_word_percentage": 0.0,
                    "weak_word_breakdown": {},
                    "total_words": 0
                }
            
            weak_words_found = []
            weak_word_breakdown = {}
            
            # Use ML pattern recognition if available (Phase 3 improvement)
            ml_patterns = None
            if ML_PATTERN_AVAILABLE and ml_pattern_recognizer:
                try:
                    ml_patterns = ml_pattern_recognizer.get_weak_word_patterns(transcript)
                    logger.info(f"ML pattern recognition found {ml_patterns.get('total_weak_instances', 0)} weak word instances")
                except Exception as e:
                    logger.warning(f"ML pattern recognition failed: {e}")
                    ml_patterns = None
            
            # Context-aware analysis if spaCy is available
            if self.nlp:
                try:
                    doc = self.nlp(transcript)
                    word_to_token = {token.text.lower().strip('.,!?;:()[]{}"\''): token for token in doc}
                    
                    for i, word in enumerate(words):
                        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                        
                        # Check ML prediction first if available
                        ml_confidence = None
                        if ml_patterns:
                            for ml_word in ml_patterns.get('weak_words_detected', []):
                                if ml_word.get('word', '').lower().strip('.,!?;:()[]{}"\'') == word_lower:
                                    ml_confidence = ml_word.get('confidence', 0.5)
                                    break
                        
                        if word_lower in self.WEAK_WORDS:
                            # Check context to see if it's actually weak in this usage
                            token = word_to_token.get(word_lower)
                            is_weak = True
                            
                            if token:
                                # Context rules for specific words
                                if word_lower == "just" and token.pos_ != "ADV":
                                    # "just" as adjective (e.g., "just cause") is not weak
                                    is_weak = False
                                elif word_lower == "really" and token.dep_ not in ["advmod", "amod"]:
                                    # "really" not modifying something is less weak
                                    is_weak = False
                                elif word_lower == "very" and token.dep_ not in ["advmod", "amod"]:
                                    # "very" not modifying is less weak
                                    is_weak = False
                                elif word_lower == "like" and token.pos_ == "ADP":
                                    # "like" as preposition (e.g., "like a bird") is not weak
                                    is_weak = False
                            
                            # ML override: if ML says it's weak with high confidence, trust it
                            if ml_confidence and ml_confidence > 0.7 and is_weak:
                                is_weak = True  # ML confirms it's weak
                            elif ml_confidence and ml_confidence < 0.4:
                                is_weak = False  # ML says it's not weak
                            
                            if is_weak:
                                weak_words_found.append({
                                    "word": word,
                                    "position": i,
                                    "suggestion": self.WEAK_WORDS[word_lower],
                                    "context": token.pos_ if token else "unknown",
                                    "ml_confidence": ml_confidence
                                })
                                weak_word_breakdown[word_lower] = weak_word_breakdown.get(word_lower, 0) + 1
                except Exception as e:
                    logger.warning(f"Context-aware analysis failed: {e}, using simple matching")
                    # Fallback to simple matching
                    for i, word in enumerate(words):
                        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                        if word_lower in self.WEAK_WORDS:
                            weak_words_found.append({
                                "word": word,
                                "position": i,
                                "suggestion": self.WEAK_WORDS[word_lower]
                            })
                            weak_word_breakdown[word_lower] = weak_word_breakdown.get(word_lower, 0) + 1
            else:
                # Simple matching without context
                for i, word in enumerate(words):
                    word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                    
                    if word_lower in self.WEAK_WORDS:
                        weak_words_found.append({
                            "word": word,
                            "position": i,
                            "suggestion": self.WEAK_WORDS[word_lower]
                        })
                        weak_word_breakdown[word_lower] = weak_word_breakdown.get(word_lower, 0) + 1
            
            weak_word_count = len(weak_words_found)
            weak_word_percentage = (weak_word_count / total_words * 100) if total_words > 0 else 0.0
            
            return {
                "weak_words_found": weak_words_found,
                "weak_word_count": weak_word_count,
                "weak_word_percentage": round(weak_word_percentage, 2),
                "weak_word_breakdown": weak_word_breakdown,
                "total_words": total_words
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weak words: {e}")
            return {
                "weak_words_found": [],
                "weak_word_count": 0,
                "weak_word_percentage": 0.0,
                "weak_word_breakdown": {},
                "total_words": 0
            }
    
    def analyze_filler_words(self, transcript: str) -> Dict:
        """
        Analyze filler words including mumbling patterns
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Dictionary with filler words analysis
        """
        try:
            if not transcript or not transcript.strip():
                return {
                    "filler_words_found": [],
                    "filler_count": 0,
                    "filler_percentage": 0.0,
                    "filler_breakdown": {},
                    "mumbling_detected": False,
                    "total_words": 0
                }
            
            words = self._tokenize(transcript)
            total_words = len(words)
            
            if total_words == 0:
                return {
                    "filler_words_found": [],
                    "filler_count": 0,
                    "filler_percentage": 0.0,
                    "filler_breakdown": {},
                    "mumbling_detected": False,
                    "total_words": 0
                }
            
            filler_words_found = []
            filler_breakdown = {}
            mumbling_patterns = ["um", "uh", "ah", "eh", "er", "hmm", "erm", "mhm", "uh-huh"]
            mumbling_count = 0
            
            for i, word in enumerate(words):
                word_lower = word.lower().strip('.,!?;:()[]{}"\'')
                
                # Check for exact matches
                if word_lower in self.EXTENDED_FILLERS:
                    filler_words_found.append({
                        "word": word,
                        "position": i,
                        "type": "mumbling" if word_lower in mumbling_patterns else "filler"
                    })
                    filler_breakdown[word_lower] = filler_breakdown.get(word_lower, 0) + 1
                    if word_lower in mumbling_patterns:
                        mumbling_count += 1
                
                # Check for stuttering patterns (repeated words)
                if i > 0:
                    prev_word = words[i-1].lower().strip('.,!?;:()[]{}"\'')
                    if word_lower == prev_word and word_lower in self.EXTENDED_FILLERS:
                        filler_words_found.append({
                            "word": f"{word} (repeated)",
                            "position": i,
                            "type": "stuttering"
                        })
            
            filler_count = len(filler_words_found)
            filler_percentage = (filler_count / total_words * 100) if total_words > 0 else 0.0
            mumbling_detected = mumbling_count > 2 or (mumbling_count / total_words > 0.02) if total_words > 0 else False
            
            return {
                "filler_words_found": filler_words_found,
                "filler_count": filler_count,
                "filler_percentage": round(filler_percentage, 2),
                "filler_breakdown": filler_breakdown,
                "mumbling_detected": mumbling_detected,
                "mumbling_count": mumbling_count,
                "total_words": total_words
            }
            
        except Exception as e:
            logger.error(f"Error analyzing filler words: {e}")
            return {
                "filler_words_found": [],
                "filler_count": 0,
                "filler_percentage": 0.0,
                "filler_breakdown": {},
                "mumbling_detected": False,
                "total_words": 0
            }
    
    def analyze_vocabulary_quality(self, transcript: str) -> Dict:
        """
        Analyze vocabulary richness and quality
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Dictionary with vocabulary analysis
        """
        try:
            if not transcript or not transcript.strip():
                return {
                    "vocabulary_richness": 0.0,
                    "unique_word_ratio": 0.0,
                    "total_words": 0,
                    "unique_words": 0,
                    "complex_words": 0,
                    "complex_word_ratio": 0.0,
                    "suggestions": []
                }
            
            words = self._tokenize(transcript)
            total_words = len(words)
            
            if total_words == 0:
                return {
                    "vocabulary_richness": 0.0,
                    "unique_word_ratio": 0.0,
                    "total_words": 0,
                    "unique_words": 0,
                    "complex_words": 0,
                    "complex_word_ratio": 0.0,
                    "suggestions": []
                }
            
            # Calculate unique words
            unique_words = len(set(word.lower().strip('.,!?;:()[]{}"\'') for word in words))
            unique_word_ratio = (unique_words / total_words) if total_words > 0 else 0.0
            
            # Calculate vocabulary richness (type-token ratio)
            vocabulary_richness = unique_word_ratio
            
            # Count complex words (more than 6 characters, excluding common fillers)
            complex_words = sum(1 for word in words 
                              if len(word.strip('.,!?;:()[]{}"\'')) > 6 
                              and word.lower().strip('.,!?;:()[]{}"\'') not in self.EXTENDED_FILLERS)
            complex_word_ratio = (complex_words / total_words) if total_words > 0 else 0.0
            
            # Generate suggestions
            suggestions = []
            if vocabulary_richness < 0.5:
                suggestions.append(self.VOCABULARY_SUGGESTIONS["low_richness"])
            if vocabulary_richness < 0.4:
                suggestions.append(self.VOCABULARY_SUGGESTIONS["high_repetition"])
            if complex_word_ratio < 0.15:
                suggestions.append(self.VOCABULARY_SUGGESTIONS["simple_words"])
            
            return {
                "vocabulary_richness": round(vocabulary_richness, 3),
                "unique_word_ratio": round(unique_word_ratio, 3),
                "total_words": total_words,
                "unique_words": unique_words,
                "complex_words": complex_words,
                "complex_word_ratio": round(complex_word_ratio, 3),
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing vocabulary quality: {e}")
            return {
                "vocabulary_richness": 0.0,
                "unique_word_ratio": 0.0,
                "total_words": 0,
                "unique_words": 0,
                "complex_words": 0,
                "complex_word_ratio": 0.0,
                "suggestions": []
            }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Split into words
        words = text.split()
        return words
    
    def get_comprehensive_analysis(self, transcript: str) -> Dict:
        """
        Get comprehensive word analysis
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Complete word analysis dictionary
        """
        try:
            weak_words_analysis = self.analyze_weak_words(transcript)
            filler_words_analysis = self.analyze_filler_words(transcript)
            vocabulary_analysis = self.analyze_vocabulary_quality(transcript)
            
            return {
                "weak_words": weak_words_analysis,
                "filler_words": filler_words_analysis,
                "vocabulary": vocabulary_analysis,
                "summary": {
                    "total_words": vocabulary_analysis.get("total_words", 0),
                    "weak_word_percentage": weak_words_analysis.get("weak_word_percentage", 0.0),
                    "filler_percentage": filler_words_analysis.get("filler_percentage", 0.0),
                    "vocabulary_richness": vocabulary_analysis.get("vocabulary_richness", 0.0),
                    "overall_quality": self._calculate_overall_quality(
                        weak_words_analysis,
                        filler_words_analysis,
                        vocabulary_analysis
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive word analysis: {e}")
            return {}

    def _calculate_overall_quality(self, weak_words: Dict, fillers: Dict, vocabulary: Dict) -> str:
        """Calculate overall word quality assessment"""
        weak_pct = weak_words.get("weak_word_percentage", 0)
        filler_pct = fillers.get("filler_percentage", 0)
        vocab_richness = vocabulary.get("vocabulary_richness", 0)
        
        if weak_pct < 2 and filler_pct < 4 and vocab_richness > 0.6:
            return "Excellent"
        elif weak_pct < 4 and filler_pct < 8 and vocab_richness > 0.5:
            return "Good"
        elif weak_pct < 8 and filler_pct < 15 and vocab_richness > 0.4:
            return "Fair"
        else:
            return "Needs Improvement"

# Global instance
word_analyzer = WordAnalyzer()



