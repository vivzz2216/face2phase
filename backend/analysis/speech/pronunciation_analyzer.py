"""
Pronunciation Analysis Module
Detects syllable stress patterns and pronunciation issues using CMU Pronouncing Dictionary
Enhanced with acoustic analysis for better accuracy
"""
import logging
import re
from typing import Dict, List, Optional
from collections import Counter, defaultdict

# Import acoustic analyzer (optional - will work without it)
try:
    from analysis.speech.acoustic_pronunciation_analyzer import acoustic_pronunciation_analyzer
    ACOUSTIC_AVAILABLE = True
except ImportError:
    acoustic_pronunciation_analyzer = None
    ACOUSTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import CMU Pronouncing Dictionary
try:
    import nltk
    try:
        from nltk.corpus import cmudict
        CMUDICT_AVAILABLE = True
    except LookupError:
        logger.warning("CMU Pronouncing Dictionary not found. Attempting to download...")
        try:
            nltk.download('cmudict', quiet=True)
            from nltk.corpus import cmudict
            CMUDICT_AVAILABLE = True
        except Exception as e:
            logger.error(f"Failed to download CMU dictionary: {e}")
            CMUDICT_AVAILABLE = False
    except ImportError:
        logger.warning("NLTK not available. Using fallback pronunciation data.")
        CMUDICT_AVAILABLE = False
except ImportError:
    logger.warning("NLTK not installed. Using fallback pronunciation data.")
    CMUDICT_AVAILABLE = False

# Common technical words with known stress patterns (fallback)
COMMON_STRESS_PATTERNS = {
    'python': ('PY-thon', '/ˈpaɪθɑːn/', 0),
    'database': ('DA-ta-base', '/ˈdeɪtəbeɪs/', 0),
    'algorithm': ('AL-go-rithm', '/ˈælɡərɪðəm/', 0),
    'application': ('ap-pli-CA-tion', '/ˌæplɪˈkeɪʃən/', 2),
    'presentation': ('pres-en-TA-tion', '/ˌprezənˈteɪʃən/', 2),
    'technology': ('tech-NOL-o-gy', '/tekˈnɒlədʒi/', 1),
    'developer': ('de-VEL-op-er', '/dɪˈveləpər/', 1),
    'computer': ('com-PU-ter', '/kəmˈpjuːtər/', 1),
    'system': ('SYS-tem', '/ˈsɪstəm/', 0),
    'project': ('PROJ-ect', '/ˈprɒdʒekt/', 0),
    'research': ('re-SEARCH', '/rɪˈsɜːrtʃ/', 1),
    'analysis': ('a-NAL-y-sis', '/əˈnæləsɪs/', 1),
    'implementation': ('im-ple-men-TA-tion', '/ˌɪmplɪmenˈteɪʃən/', 2),
    'architecture': ('AR-chi-tec-ture', '/ˈɑːrkɪtektʃər/', 0),
    'framework': ('FRAME-work', '/ˈfreɪmwɜːrk/', 0),
}

class PronunciationAnalyzer:
    """Analyzes pronunciation and detects syllable stress issues"""
    
    def __init__(self):
        self.cmudict_dict = None
        if CMUDICT_AVAILABLE:
            try:
                self.cmudict_dict = cmudict.dict()
                logger.info("CMU Pronouncing Dictionary loaded successfully")
            except Exception as e:
                logger.error(f"Error loading CMU dictionary: {e}")
                self.cmudict_dict = None
    
    def get_pronunciation(self, word: str) -> Optional[List[str]]:
        """
        Get pronunciation from CMU dictionary
        
        Args:
            word: Word to look up
            
        Returns:
            List of pronunciation phonemes or None
        """
        if not self.cmudict_dict:
            return None
        
        word_lower = word.lower().strip('.,!?;:()[]{}"\'')
        return self.cmudict_dict.get(word_lower)
    
    def get_stress_pattern(self, pronunciation: List[str]) -> tuple:
        """
        Extract stress pattern from pronunciation
        
        Args:
            pronunciation: List of phonemes from CMU dictionary
            
        Returns:
            Tuple of (primary_stress_index, secondary_stress_indices, syllables)
        """
        primary_stress = None
        secondary_stresses = []
        syllables = []
        current_syllable = []
        
        for phoneme in pronunciation:
            # Check for stress markers (0=no stress, 1=primary, 2=secondary)
            if phoneme.endswith('1'):
                primary_stress = len(syllables)
                current_syllable.append(phoneme.rstrip('1'))
                syllables.append(current_syllable)
                current_syllable = []
            elif phoneme.endswith('2'):
                secondary_stresses.append(len(syllables))
                current_syllable.append(phoneme.rstrip('2'))
                syllables.append(current_syllable)
                current_syllable = []
            elif phoneme.endswith('0'):
                current_syllable.append(phoneme.rstrip('0'))
                syllables.append(current_syllable)
                current_syllable = []
            else:
                current_syllable.append(phoneme)
        
        if current_syllable:
            syllables.append(current_syllable)
        
        return primary_stress, secondary_stresses, syllables
    
    def format_stress_pattern(self, word: str, primary_stress: int, syllables: List[List[str]]) -> str:
        """
        Format word with stress marks
        
        Args:
            word: Original word
            primary_stress: Index of primary stressed syllable
            syllables: List of syllable phonemes
            
        Returns:
            Formatted word with uppercase stress (e.g., "PY-thon")
        """
        if primary_stress is None or not syllables:
            return word
        
        # Simple heuristic: split word into approximate syllables
        word_clean = word.lower().strip('.,!?;:()[]{}"\'')
        vowels = 'aeiou'
        
        # Count vowels to estimate syllables
        vowel_positions = [i for i, char in enumerate(word_clean) if char in vowels]
        
        if not vowel_positions:
            return word
        
        # Estimate syllable boundaries
        syllable_count = len(vowel_positions)
        if syllable_count <= primary_stress:
            return word
        
        # Create stress pattern
        parts = []
        for i, pos in enumerate(vowel_positions):
            if i == primary_stress:
                # This syllable is stressed - make uppercase
                start = vowel_positions[i-1] + 1 if i > 0 else 0
                end = vowel_positions[i+1] if i < len(vowel_positions) - 1 else len(word_clean)
                parts.append(word_clean[start:end].upper())
            else:
                start = vowel_positions[i-1] + 1 if i > 0 else 0
                end = vowel_positions[i+1] if i < len(vowel_positions) - 1 else len(word_clean)
                parts.append(word_clean[start:end].lower())
        
        return '-'.join(parts)
    
    def get_ipa_notation(self, pronunciation: List[str]) -> str:
        """
        Convert CMU pronunciation to IPA notation (simplified)
        
        Args:
            pronunciation: List of phonemes
            
        Returns:
            IPA notation string
        """
        # Simplified CMU to IPA mapping for common patterns
        cmu_to_ipa = {
            'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
            'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɜr',
            'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
            'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
            'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
            'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
            'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
        }
        
        ipa_parts = []
        primary_stress_idx = None
        
        for i, phoneme in enumerate(pronunciation):
            if phoneme.endswith('1'):
                primary_stress_idx = len(ipa_parts)
                base = phoneme.rstrip('1')
                ipa_parts.append(cmu_to_ipa.get(base, base))
            elif phoneme.endswith('2'):
                base = phoneme.rstrip('2')
                ipa_parts.append(cmu_to_ipa.get(base, base))
            elif phoneme.endswith('0'):
                base = phoneme.rstrip('0')
                ipa_parts.append(cmu_to_ipa.get(base, base))
            else:
                ipa_parts.append(cmu_to_ipa.get(phoneme, phoneme))
        
        # Build IPA string with stress mark
        ipa_str = ''.join(ipa_parts)
        
        # Add primary stress mark (ˈ) before stressed syllable if found
        if primary_stress_idx is not None and primary_stress_idx < len(ipa_parts):
            # Insert stress mark before the stressed syllable
            # Simplified: add at start if first syllable, otherwise approximate position
            if primary_stress_idx == 0:
                ipa_str = 'ˈ' + ipa_str
            else:
                # Approximate: add stress mark early in the word
                ipa_str = 'ˈ' + ipa_str
        
        return '/' + ipa_str + '/'
    
    def analyze_word(self, word: str, context: str = "") -> Optional[Dict]:
        """
        Analyze a single word for pronunciation issues
        
        Args:
            word: Word to analyze
            context: Context sentence where word appears
            
        Returns:
            Dictionary with pronunciation analysis or None
        """
        word_clean = word.lower().strip('.,!?;:()[]{}"\'')
        
        # Check common patterns first
        if word_clean in COMMON_STRESS_PATTERNS:
            pattern, ipa, _ = COMMON_STRESS_PATTERNS[word_clean]
            # Generate incorrect pattern for common words
            incorrect_pattern = self._generate_incorrect_stress_for_common(word_clean, pattern)
            return {
                'word': word,
                'incorrect': incorrect_pattern,
                'correct': pattern,
                'phonetic': ipa,
                'context': context,
                'source': 'common_patterns'
            }
        
        # Try CMU dictionary
        pronunciation = self.get_pronunciation(word_clean)
        if pronunciation and len(pronunciation) > 0:
            primary_stress, secondary_stresses, syllables = self.get_stress_pattern(pronunciation[0])
            
            if primary_stress is not None:
                correct_pattern = self.format_stress_pattern(word, primary_stress, syllables)
                ipa = self.get_ipa_notation(pronunciation[0])
                
                # Generate incorrect pattern (wrong stress)
                incorrect_pattern = self._generate_incorrect_stress(word, primary_stress, syllables)
                
                return {
                    'word': word,
                    'incorrect': incorrect_pattern,
                    'correct': correct_pattern,
                    'phonetic': ipa,
                    'context': context,
                    'source': 'cmudict'
                }
        
        return None
    
    def _generate_incorrect_stress_for_common(self, word: str, correct_pattern: str) -> str:
        """Generate incorrect stress pattern for common words"""
        # Common mistake: stress on wrong syllable
        # For most technical words, people stress the last syllable instead of first
        word_lower = word.lower()
        vowels = 'aeiou'
        vowel_positions = [i for i, char in enumerate(word_lower) if char in vowels]
        
        if len(vowel_positions) <= 1:
            return word_lower
        
        # Generate incorrect: stress on last syllable
        parts = []
        for i, pos in enumerate(vowel_positions):
            start = vowel_positions[i-1] + 1 if i > 0 else 0
            end = vowel_positions[i+1] if i < len(vowel_positions) - 1 else len(word_lower)
            syllable = word_lower[start:end]
            
            if i == len(vowel_positions) - 1:
                # Last syllable - uppercase (incorrect stress)
                parts.append(syllable.upper())
            else:
                parts.append(syllable.lower())
        
        return '-'.join(parts)
    
    def _generate_incorrect_stress(self, word: str, correct_stress: int, syllables: List[List[str]]) -> str:
        """
        Generate likely incorrect stress pattern
        
        Args:
            word: Original word
            correct_stress: Index of correct stress
            syllables: Syllable structure
            
        Returns:
            Incorrect stress pattern (common mistake)
        """
        word_clean = word.lower().strip('.,!?;:()[]{}"\'')
        vowels = 'aeiou'
        vowel_positions = [i for i, char in enumerate(word_clean) if char in vowels]
        
        if not vowel_positions or len(vowel_positions) <= 1:
            return word_clean
        
        # Common mistake: stress on last syllable instead of first
        if correct_stress == 0 and len(vowel_positions) > 1:
            # Show incorrect: stress on last syllable
            parts = []
            for i, pos in enumerate(vowel_positions):
                start = vowel_positions[i-1] + 1 if i > 0 else 0
                end = vowel_positions[i+1] if i < len(vowel_positions) - 1 else len(word_clean)
                syllable = word_clean[start:end]
                
                if i == len(vowel_positions) - 1:
                    # Last syllable - uppercase (incorrect stress)
                    parts.append(syllable.upper())
                else:
                    parts.append(syllable.lower())
            return '-'.join(parts)
        
        # If stress is on middle/last, show incorrect on first
        if correct_stress > 0 and len(vowel_positions) > 1:
            parts = []
            for i, pos in enumerate(vowel_positions):
                start = vowel_positions[i-1] + 1 if i > 0 else 0
                end = vowel_positions[i+1] if i < len(vowel_positions) - 1 else len(word_clean)
                syllable = word_clean[start:end]
                
                if i == 0:
                    # First syllable - uppercase (incorrect stress)
                    parts.append(syllable.upper())
                else:
                    parts.append(syllable.lower())
            return '-'.join(parts)
        
        # Default: show word in lowercase as "incorrect"
        return word_clean.lower()
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """
        Analyze entire transcript for pronunciation issues
        
        Args:
            transcript: Full transcript text
            
        Returns:
            Dictionary with pronunciation issues found
        """
        try:
            # Common technical/professional words to check
            important_words = [
                'python', 'database', 'algorithm', 'application', 'presentation',
                'technology', 'developer', 'computer', 'system', 'project',
                'research', 'analysis', 'implementation', 'architecture', 'framework',
                'mysql', 'javascript', 'react', 'angular', 'nodejs', 'api', 'http',
                'server', 'client', 'backend', 'frontend', 'software', 'hardware'
            ]
            
            issues = []
            words = re.findall(r'\b\w+\b', transcript.lower())
            word_frequencies = Counter(words)
            word_context_map: Dict[str, str] = {}
            
            sentences = re.split(r'[.!?]+', transcript)
            for sentence in sentences:
                sentence_stripped = sentence.strip()
                if not sentence_stripped:
                    continue
                for word in re.findall(r'\b\w+\b', sentence_stripped):
                    lower = word.lower()
                    if lower not in word_context_map:
                        word_context_map[lower] = sentence_stripped
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                for important_word in important_words:
                    if important_word in sentence_lower:
                        # Find the actual word in context (case-sensitive)
                        word_pattern = re.compile(r'\b' + re.escape(important_word) + r'\b', re.IGNORECASE)
                        matches = word_pattern.finditer(sentence)
                        for match in matches:
                            word = match.group()
                            analysis = self.analyze_word(word, sentence.strip())
                            if analysis:
                                # Check if we already have this word
                                if not any(issue['word'].lower() == word.lower() for issue in issues):
                                    issues.append(analysis)
            
            analyzed_words = {issue['word'].lower() for issue in issues}
            
            # Broaden coverage: analyse frequent multi-syllable words the speaker used
            vowels = set('aeiou')
            stopwords = {
                'the', 'and', 'but', 'with', 'from', 'that', 'this', 'there',
                'have', 'will', 'would', 'should', 'could', 'about', 'into',
                'over', 'under', 'after', 'before', 'still', 'really', 'very',
                'people', 'because', 'while', 'where', 'when', 'which', 'those',
                'these', 'their', 'they', 'them', 'your', 'yours', 'ours', 'were',
                'been', 'being', 'again', 'through', 'every', 'something', 'nothing'
            }
            
            candidate_words = []
            for word, count in word_frequencies.items():
                if len(word) <= 3 or word.isdigit():
                    continue
                if word in stopwords:
                    continue
                if word in analyzed_words:
                    continue
                syllable_guess = sum(1 for ch in word if ch in vowels)
                if syllable_guess < 2 and len(word) < 7:
                    continue
                candidate_words.append((word, count))
            
            candidate_words.sort(key=lambda item: (-item[1], -len(item[0])))
            
            for word, _ in candidate_words[:20]:
                original_word = None
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                match = pattern.search(transcript)
                if match:
                    original_word = match.group()
                else:
                    original_word = word
                context_sentence = word_context_map.get(word, '')
                analysis = self.analyze_word(original_word, context_sentence)
                if analysis and analysis['word'].lower() not in analyzed_words:
                    issues.append(analysis)
                    analyzed_words.add(analysis['word'].lower())
                    if len(issues) >= 12:
                        break

            # Limit final list to most actionable items
            issues = issues[:10]
            
            return {
                'issues': issues,
                'total_issues': len(issues),
                'general_advice': self._generate_general_advice(issues)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing transcript for pronunciation: {e}")
            return {
                'issues': [],
                'total_issues': 0,
                'general_advice': 'Pronunciation analysis unavailable.'
            }
    
    def _generate_general_advice(self, issues: List[Dict]) -> str:
        """Generate general pronunciation advice based on issues found"""
        if not issues:
            return "Your pronunciation is generally good. Continue practicing technical terms for clarity."
        
        focus_terms = [issue.get('word') for issue in issues if issue.get('word')]
        focus_terms = [term for term in focus_terms if isinstance(term, str)]
        top_terms = ', '.join(focus_terms[:3])
        
        if len(issues) == 1:
            return f"Focus on the stress pattern in \"{focus_terms[0]}\". Record yourself and compare with a dictionary model until the stress matches."
        elif len(issues) <= 3:
            return f"Pay attention to syllable stress in terms such as {top_terms}. Slow down on multi-syllable words, emphasising the uppercase syllables shown."
        else:
            return f"Work on overall syllable stress accuracy. Prioritise words like {top_terms} and rehearse them with a pronunciation coach or IPA guide."

# Global instance
pronunciation_analyzer = PronunciationAnalyzer()

