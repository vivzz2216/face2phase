"""
Text/NLP analysis module for vocabulary and expression scoring
"""
import logging
from typing import Dict, List, Tuple, Optional
import re
import gzip
from collections import Counter
import numpy as np
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
from ...core.settings import FILLER_WORDS, OPENAI_API_KEY, USE_OPENAI_API
from ...utils.device_detector import device_manager

# Import OpenAI enhancer
try:
    from ...services.openai_enhancer import OpenAIEnhancer
    OPENAI_ENHANCER_AVAILABLE = True
except ImportError:
    OPENAI_ENHANCER_AVAILABLE = False

logger = logging.getLogger(__name__)

BASIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "for", "in", "on",
    "at", "with", "by", "from", "as", "is", "are", "was", "were", "be", "being",
    "been", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "they", "we", "me", "him", "her", "them", "my", "your", "his", "their",
    "our", "mine", "yours", "hers", "ours", "theirs", "there", "here", "then",
    "than", "so", "because", "very", "really", "just", "like", "also", "well",
    "uh", "um", "uhh", "umm", "basically", "actually", "literally"
}

class TextAnalyzer:
    """Analyzes text for vocabulary richness and expression quality"""
    
    def __init__(self):
        self.spacy_model = device_manager.get_spacy_model()
        if not self.spacy_model:
            logger.warning("spaCy model not available. Text analysis will be limited.")
        
        # Initialize OpenAI enhancer if available
        if OPENAI_ENHANCER_AVAILABLE:
            self.openai_enhancer = OpenAIEnhancer()
        else:
            self.openai_enhancer = None
            logger.warning("OpenAI enhancer not available.")

    def _compute_compression_ratio(self, payload: str) -> float:
        """Return gzip compression ratio for the given payload."""
        if not payload:
            return 1.0
        data = payload.encode("utf-8", errors="ignore")
        if not data:
            return 1.0
        try:
            compressed = gzip.compress(data, compresslevel=6)
            return round(len(compressed) / max(1, len(data)), 3)
        except Exception as exc:
            logger.debug(f"Compression ratio failed: {exc}")
            return 1.0

    def _compute_distinct_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """Compute distinct-n metrics for n=1 and n=2."""
        metrics = {"distinct_1": 0.0, "distinct_2": 0.0}
        if not tokens:
            return metrics

        unique_unigrams = len(set(tokens))
        metrics["distinct_1"] = round(unique_unigrams / len(tokens), 3)

        if len(tokens) >= 2:
            bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
            unique_bigrams = len(set(bigrams))
            metrics["distinct_2"] = round(unique_bigrams / len(bigrams), 3)
        return metrics

    def _compute_repeated_ngram_pct(self, tokens: List[str], n: int = 4) -> float:
        """Percentage of n-grams that repeat at least once."""
        if len(tokens) < n or n <= 0:
            return 0.0
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(count for count in counts.values() if count > 1)
        return round((repeated / len(ngrams)) * 100, 2)

    def _extract_subject_verb_pairs_doc(self, doc) -> Dict[str, any]:
        """Extract subject-verb combinations using spaCy dependency parses."""
        pairs: Counter = Counter()
        examples: Dict[str, str] = {}
        for sent in doc.sents:
            seen_in_sentence = set()
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head is not None:
                    head = token.head
                    if head.pos_ in ("VERB", "AUX"):
                        subject = token.lemma_.lower().strip()
                        verb = head.lemma_.lower().strip()
                        if subject and verb:
                            key = f"{subject}->{verb}"
                            if key not in seen_in_sentence:
                                pairs[key] += 1
                                examples.setdefault(key, sent.text.strip())
                                seen_in_sentence.add(key)
        total = sum(pairs.values())
        unique = len(pairs)
        diversity = (unique / total) if total else 0.0
        top_pairs = [{"pair": pair, "count": count, "example": examples.get(pair, "")} for pair, count in pairs.most_common(6)]
        return {
            "total_pairs": total,
            "unique_pairs": unique,
            "diversity": round(diversity, 3),
            "top_pairs": top_pairs
        }

    def _heuristic_subject_verb_pairs(self, sentences: List[str]) -> Dict[str, any]:
        """Fallback heuristic to approximate subject-verb combinations without spaCy."""
        token_pattern = re.compile(r"[A-Za-z']+")
        pairs: Counter = Counter()
        examples: Dict[str, str] = {}
        for sentence in sentences:
            tokens = [tok.lower() for tok in token_pattern.findall(sentence)]
            if len(tokens) < 2:
                continue
            subject = tokens[0]
            verb = tokens[1]
            pair = f"{subject}->{verb}"
            pairs[pair] += 1
            examples.setdefault(pair, sentence.strip())
        total = sum(pairs.values())
        unique = len(pairs)
        diversity = (unique / total) if total else 0.0
        top_pairs = [{"pair": pair, "count": count, "example": examples.get(pair, "")} for pair, count in pairs.most_common(6)]
        return {
            "total_pairs": total,
            "unique_pairs": unique,
            "diversity": round(diversity, 3),
            "top_pairs": top_pairs
        }

    def _compute_subject_verb_stats(self, text: str, sentences: List[str]) -> Dict[str, any]:
        """Compute subject-verb combination diversity metrics."""
        if self.spacy_model:
            try:
                doc = self.spacy_model(text)
                return self._extract_subject_verb_pairs_doc(doc)
            except Exception as exc:
                logger.debug(f"spaCy subject-verb extraction failed: {exc}")
        return self._heuristic_subject_verb_pairs(sentences)
    
    def _fallback_key_topics(self, text: str, top_n: int = 10) -> Tuple[List[str], Dict[str, int]]:
        """Generate key topics using a simple frequency approach when spaCy is unavailable."""
        clean_text = self.preprocess_text(text)
        tokens = [token for token in re.split(r'\W+', clean_text.lower()) if token]
        filtered_tokens = [
            token for token in tokens
            if len(token) > 3 and token not in BASIC_STOPWORDS and not token.isdigit()
        ]
        if not filtered_tokens:
            return [], {}
        token_counts = Counter(filtered_tokens)
        top_keywords = [token for token, _ in token_counts.most_common(top_n)]
        return top_keywords, dict(token_counts.most_common(top_n))
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?;:]', '', text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def analyze_vocabulary_richness(self, text: str) -> Dict:
        """
        Analyze vocabulary richness and diversity
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing vocabulary metrics
        """
        try:
            if not text.strip():
                return {
                    'total_words': 0,
                    'unique_words': 0,
                    'vocabulary_richness': 0,
                    'type_token_ratio': 0,
                    'avg_word_length': 0,
                    'complex_words': 0,
                    'complex_word_ratio': 0
                }
            
            # Clean text
            clean_text = self.preprocess_text(text)
            words = clean_text.split()
            
            if not words:
                return {
                    'total_words': 0,
                    'unique_words': 0,
                    'vocabulary_richness': 0,
                    'type_token_ratio': 0,
                    'avg_word_length': 0,
                    'complex_words': 0,
                    'complex_word_ratio': 0
                }
            
            # Basic metrics
            total_words = len(words)
            unique_words = len(set(words))
            vocabulary_richness = unique_words / total_words if total_words > 0 else 0
            
            # Type-token ratio (more sophisticated than simple richness)
            word_counts = Counter(words)
            type_token_ratio = len(word_counts) / total_words if total_words > 0 else 0
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
            
            # Complex words (words with more than 6 characters)
            complex_words = sum(1 for word in words if len(word) > 6)
            complex_word_ratio = complex_words / total_words if total_words > 0 else 0
            
            return {
                'total_words': total_words,
                'unique_words': unique_words,
                'vocabulary_richness': vocabulary_richness,
                'type_token_ratio': type_token_ratio,
                'avg_word_length': avg_word_length,
                'complex_words': complex_words,
                'complex_word_ratio': complex_word_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing vocabulary richness: {e}")
            return {}
    
    def analyze_sentence_structure(self, text: str) -> Dict:
        """
        Analyze sentence structure and complexity
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing sentence structure metrics
        """
        try:
            if not text.strip():
                return {
                    'total_sentences': 0,
                    'avg_sentence_length': 0,
                    'sentence_variety': 0,
                    'complex_sentences': 0,
                    'complex_sentence_ratio': 0,
                    'sentence_openers': {},
                    'opener_diversity': 0.0,
                    'weak_openers': {},
                    'opener_examples': {},
                    'subject_verb_stats': {},
                    'subject_verb_diversity': 0.0
                }
            
            # Split into sentences (simple approach)
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {
                    'total_sentences': 0,
                    'avg_sentence_length': 0,
                    'sentence_variety': 0,
                    'complex_sentences': 0,
                    'complex_sentence_ratio': 0,
                    'sentence_openers': {},
                    'opener_diversity': 0.0,
                    'weak_openers': {},
                    'opener_examples': {},
                    'subject_verb_stats': {},
                    'subject_verb_diversity': 0.0
                }
            
            # Calculate metrics
            total_sentences = len(sentences)
            sentence_lengths = [len(sentence.split()) for sentence in sentences]
            avg_sentence_length = sum(sentence_lengths) / total_sentences if total_sentences > 0 else 0
            
            # Sentence variety (standard deviation of sentence lengths)
            if len(sentence_lengths) > 1:
                mean_length = avg_sentence_length
                variance = sum((length - mean_length) ** 2 for length in sentence_lengths) / (len(sentence_lengths) - 1)
                sentence_variety = variance ** 0.5
            else:
                sentence_variety = 0
            
            # Complex sentences (sentences with more than 15 words)
            complex_sentences = sum(1 for length in sentence_lengths if length > 15)
            complex_sentence_ratio = complex_sentences / total_sentences if total_sentences > 0 else 0
            
            # ADVANCED: Analyze sentence openers
            opener_analysis = self.analyze_sentence_openers(sentences)
            subject_verb_stats = self._compute_subject_verb_stats(text, sentences)
            
            return {
                'total_sentences': total_sentences,
                'avg_sentence_length': avg_sentence_length,
                'sentence_variety': sentence_variety,
                'complex_sentences': complex_sentences,
                'complex_sentence_ratio': complex_sentence_ratio,
                'sentence_openers': opener_analysis,
                'opener_diversity': opener_analysis.get('opener_diversity', 0),
                'weak_openers': opener_analysis.get('weak_openers', {}),
                'opener_examples': opener_analysis.get('opener_examples', {}),
                'subject_verb_stats': subject_verb_stats,
                'subject_verb_diversity': subject_verb_stats.get('diversity', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentence structure: {e}")
            return {
                'total_sentences': 0,
                'avg_sentence_length': 0,
                'sentence_variety': 0,
                'complex_sentences': 0,
                'complex_sentence_ratio': 0,
                'sentence_openers': {},
                'opener_diversity': 0.0,
                'weak_openers': {},
                'opener_examples': {},
                'subject_verb_stats': {},
                'subject_verb_diversity': 0.0
            }
    
    def analyze_sentence_openers(self, sentences: List[str]) -> Dict:
        """
        ADVANCED: Analyze sentence openers to detect weak patterns
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dictionary with opener analysis
        """
        try:
            if not sentences:
                return {
                    'total_sentences': 0,
                    'weak_openers': {},
                    'opener_percentages': {},
                    'opener_diversity': 0,
                    'recommendations': []
                }
            
            # Weak sentence opener patterns (professional writing standards)
            weak_opener_patterns = {
                'it': r'^\s*it\s+',
                'there': r'^\s*there\s+(is|are|was|were|will|would)\s+',
                'this': r'^\s*this\s+',
                'that': r'^\s*that\s+',
                'i': r'^\s*i\s+',
                'we': r'^\s*we\s+',
                'you': r'^\s*you\s+',
                'the': r'^\s*the\s+',
                'a': r'^\s*a\s+',
                'an': r'^\s*an\s+',
                'so': r'^\s*so\s+',
                'but': r'^\s*but\s+',
                'and': r'^\s*and\s+',
                'because': r'^\s*because\s+',
                'well': r'^\s*well\s+',
                'now': r'^\s*now\s+',
                'then': r'^\s*then\s+',
                'actually': r'^\s*actually\s+',
                'basically': r'^\s*basically\s+',
                'like': r'^\s*like\s+',
                'you_know': r'^\s*you\s+know\s+',
            }
            
            opener_counts = {}
            opener_examples = {}
            total_sentences = len(sentences)
            
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if not sentence_lower:
                    continue
                
                # Check each weak opener pattern
                for opener_name, pattern in weak_opener_patterns.items():
                    if re.match(pattern, sentence_lower, re.IGNORECASE):
                        opener_counts[opener_name] = opener_counts.get(opener_name, 0) + 1
                        # Store example (first 50 chars)
                        if opener_name not in opener_examples:
                            opener_examples[opener_name] = sentence[:50] + ('...' if len(sentence) > 50 else '')
                        break
            
            # Calculate percentages
            opener_percentages = {
                opener: round((count / total_sentences) * 100, 1) 
                for opener, count in opener_counts.items()
            }
            
            # Calculate opener diversity (how many different openers used)
            opener_diversity = len(opener_counts) / total_sentences if total_sentences > 0 else 0
            
            # Generate recommendations
            recommendations = []
            if opener_percentages.get('it', 0) > 20:
                recommendations.append(f"Reduce 'It' openers ({opener_percentages.get('it', 0)}%) - use more specific subjects")
            if opener_percentages.get('there', 0) > 15:
                recommendations.append(f"Reduce 'There is/are' constructions ({opener_percentages.get('there', 0)}%) - be more direct")
            if opener_percentages.get('this', 0) > 15:
                recommendations.append(f"Reduce 'This' openers ({opener_percentages.get('this', 0)}%) - specify what 'this' refers to")
            if opener_percentages.get('i', 0) > 30:
                recommendations.append(f"Reduce 'I' openers ({opener_percentages.get('i', 0)}%) - vary sentence structure")
            if opener_diversity < 0.3:
                recommendations.append("Increase sentence opener variety - use different starting words")
            
            return {
                'total_sentences': total_sentences,
                'weak_openers': opener_counts,
                'opener_percentages': opener_percentages,
                'opener_examples': opener_examples,
                'opener_diversity': round(opener_diversity, 3),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentence openers: {e}")
            return {
                'total_sentences': 0,
                'weak_openers': {},
                'opener_percentages': {},
                'opener_diversity': 0,
                'recommendations': []
            }
    
    def analyze_content_quality(self, text: str) -> Dict:
        """
        Analyze content quality using spaCy if available
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing content quality metrics
        """
        try:
            if not text.strip():
                return {
                    'named_entities': 0,
                    'pos_diversity': 0,
                    'content_coherence': 0,
                    'key_topics': [],
                    'keyword_counts': {}
                }

            if not self.spacy_model:
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                clean_text = self.preprocess_text(text)
                tokens = clean_text.split()
                unique_tokens = set(tokens)
                pos_diversity = len(unique_tokens) / len(tokens) if tokens else 0
                key_topics, keyword_counts = self._fallback_key_topics(text)
                topic_ratio = min(1.0, len(key_topics) / max(1, len(sentences))) if sentences else 0
                content_coherence = 0.4 + topic_ratio * 0.6 if sentences else 0
                return {
                    'named_entities': 0,
                    'pos_diversity': round(pos_diversity, 3),
                    'content_coherence': round(content_coherence, 3),
                    'key_topics': key_topics,
                    'keyword_counts': keyword_counts
                }

            doc = self.spacy_model(text)

            named_entities = len(doc.ents)
            pos_tags = [token.pos_ for token in doc if not token.is_space]
            pos_counts = Counter(pos_tags)
            pos_diversity = len(pos_counts) / len(pos_tags) if pos_tags else 0

            key_topics = []
            keyword_counter = Counter()
            for chunk in doc.noun_chunks:
                normalized = chunk.text.lower().strip()
                if not normalized:
                    continue
                if len(normalized.split()) > 5:
                    continue
                if all(token in BASIC_STOPWORDS for token in normalized.split()):
                    continue
                key_topics.append(normalized)
                keyword_counter[normalized] += 1

            key_topics = list(dict.fromkeys(key_topics))[:10]
            keyword_counts = dict(keyword_counter.most_common(10)) or dict(Counter(key_topics).most_common(10))

            sentences = [sent.text for sent in doc.sents if len(sent.text.strip()) > 10]
            content_coherence = min(1.0, len(sentences) / 6) if sentences else 0

            return {
                'named_entities': named_entities,
                'pos_diversity': round(pos_diversity, 3),
                'content_coherence': round(content_coherence, 3),
                'key_topics': key_topics,
                'keyword_counts': keyword_counts
            }

        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            return {
                'named_entities': 0,
                'pos_diversity': 0,
                'content_coherence': 0,
                'key_topics': [],
                'keyword_counts': {}
            }
    
    def generate_text_summary(self, text: str) -> str:
        """
        Generate a simple summary of the text
        
        Args:
            text: Input text
            
        Returns:
            Text summary
        """
        try:
            if not text.strip():
                return "No content to summarize."
            
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences by word count and position
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                word_count = len(sentence.split())
                position_score = 1.0 - (i / len(sentences))  # Earlier sentences get higher score
                score = word_count * position_score
                scored_sentences.append((score, sentence))
            
            # Select top sentences
            scored_sentences.sort(reverse=True)
            summary_sentences = [sentence for _, sentence in scored_sentences[:3]]
            
            return '. '.join(summary_sentences) + '.'
            
        except Exception as e:
            logger.error(f"Error generating text summary: {e}")
            return "Summary generation failed."
    
    def calculate_vocabulary_score(self, vocabulary_metrics: Dict, structure_metrics: Dict, content_metrics: Dict) -> float:
        """
        Calculate vocabulary and expression score - STRICT REALISTIC ALGORITHM
        Based on research: Most presentations score 60-85, perfect 100 is rare
        
        Args:
            vocabulary_metrics: Vocabulary richness metrics
            structure_metrics: Sentence structure metrics
            content_metrics: Content quality metrics
            
        Returns:
            Vocabulary score
        """
        try:
            # Start with moderate base score (50 = average presentation)
            score = 50.0
            
            # === POSITIVE FACTORS ===
            
            # Vocabulary richness (STRICT scoring - 0-25 points)
            vocab_richness = vocabulary_metrics.get('vocabulary_richness', 0)
            type_token_ratio = vocabulary_metrics.get('type_token_ratio', 0)
            complex_word_ratio = vocabulary_metrics.get('complex_word_ratio', 0)
            total_words = vocabulary_metrics.get('total_words', 0)
            
            # Reward based on vocabulary richness (STRICT)
            if vocab_richness >= 0.8:
                score += 25  # Excellent diversity (very rare)
            elif vocab_richness >= 0.6:
                score += 18  # Very good diversity
            elif vocab_richness >= 0.5:
                score += 12  # Good diversity
            elif vocab_richness >= 0.4:
                score += 6   # Adequate diversity
            elif vocab_richness >= 0.3:
                score += 2   # Below average
            else:
                score -= 5   # Poor diversity
            
            # Reward for good type-token ratio (0-15 points)
            if type_token_ratio >= 0.7:
                score += 15  # Excellent
            elif type_token_ratio >= 0.6:
                score += 12
            elif type_token_ratio >= 0.5:
                score += 8
            elif type_token_ratio >= 0.4:
                score += 4
            else:
                score -= 5  # Poor type-token ratio
            
            # Reward appropriate complex word usage (0-10 points)
            if 0.20 <= complex_word_ratio <= 0.30:
                score += 10  # Ideal range
            elif 0.15 <= complex_word_ratio < 0.20 or 0.30 < complex_word_ratio <= 0.40:
                score += 6   # Good range
            elif 0.10 <= complex_word_ratio < 0.15 or 0.40 < complex_word_ratio <= 0.50:
                score += 2   # Acceptable
            elif complex_word_ratio < 0.10:
                score -= 5   # Too simple
            else:
                score -= 10  # Too complex (pretentious)
            
            # Length bonus (moderate) - 0-5 points
            if total_words > 300:
                score += 5
            elif total_words > 200:
                score += 3
            elif total_words > 100:
                score += 1
            elif total_words < 50:
                score -= 10  # Too short
            
            # Sentence structure
            avg_sentence_length = structure_metrics.get('avg_sentence_length', 0)
            sentence_variety = structure_metrics.get('sentence_variety', 0)
            total_sentences = structure_metrics.get('total_sentences', 0)
            
            # Reward good sentence length
            if 12 <= avg_sentence_length <= 18:
                score += 10  # Ideal range
            elif 8 <= avg_sentence_length < 12 or 18 < avg_sentence_length <= 25:
                score += 5   # Acceptable range
            
            # Reward sentence variety
            if sentence_variety > 5:
                score += 10  # Good variety
            elif sentence_variety > 3:
                score += 5   # Some variety
            
            # Bonus for multiple sentences (structured speech)
            if total_sentences > 5:
                score += 5
            
            # Content quality
            pos_diversity = content_metrics.get('pos_diversity', 0)
            content_coherence = content_metrics.get('content_coherence', 0)
            
            # Reward part-of-speech diversity
            if pos_diversity > 0.4:
                score += 10
            elif pos_diversity > 0.3:
                score += 5
            
            # Reward content coherence
            score += content_coherence * 15  # Increased weight
            
            # === MINIMAL PENALTIES ===
            
            # Only penalize if vocabulary is very poor
            if vocab_richness < 0.3:
                score -= 10
            
            # Only penalize very short or very long sentences
            if avg_sentence_length < 5:
                score -= 10  # Too choppy
            elif avg_sentence_length > 35:
                score -= 10  # Too long and complex
            
            # === FINAL CALCULATION ===
            # Cap the score realistically: 85+ is exceptional, 100 is perfect
            final_score = max(30, min(92, score))  # 30-92 range (100 is unattainable in practice)
            
            logger.info(f"Vocabulary score breakdown: base={50}, vocab_richness={vocab_richness:.3f}, type_token={type_token_ratio:.3f}, complex={complex_word_ratio:.3f}, sent_len={avg_sentence_length:.1f}, final={final_score:.1f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating vocabulary score: {e}")
            return 50.0  # Return realistic average instead of 70

    def _compute_simple_text_metrics(self, text: str, total_words: int) -> Dict:
        """
        Fallback analytics used when spaCy vectors are unavailable.
        Generates keyword coverage, topic coherence estimates, and sentence stats.
        """
        sentences_raw = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        token_pattern = re.compile(r'\b\w+\b')

        token_counts: Counter = Counter()
        token_context: Dict[str, str] = {}
        sentence_sets: List[set] = []
        sentence_lengths: List[int] = []
        first_word_counts: Counter = Counter()
        first_word_context: Dict[str, str] = {}
        all_tokens: List[str] = []

        for sentence in sentences_raw:
            tokens = [tok.lower() for tok in token_pattern.findall(sentence)]
            filtered = [
                tok for tok in tokens
                if tok not in BASIC_STOPWORDS and len(tok) > 2 and not tok.isdigit()
            ]

            if filtered:
                sentence_sets.append(set(filtered))
                for tok in filtered:
                    token_counts[tok] += 1
                    token_context.setdefault(tok, sentence)

            if tokens:
                sentence_lengths.append(len(tokens))
                all_tokens.extend(tokens)
                first_word = tokens[0]
                first_word_counts[first_word] += 1
                first_word_context.setdefault(first_word, sentence)

        top_topics = [word for word, _ in token_counts.most_common(8)]
        total_keywords = sum(token_counts.values())
        keyword_density = round((total_keywords / max(total_words, 1)) * 100, 2)
        coverage_ratio = round(len(set(top_topics)) / max(1, len(sentences_raw)), 3) if sentences_raw else 0.0
        keyword_details = [
            {
                "word": word,
                "count": int(token_counts[word]),
                "example": token_context.get(word, "")
            }
            for word, _ in token_counts.most_common(6)
        ]
        keyword_coverage = {
            "total_keywords": total_keywords,
            "top_keywords": top_topics[:6],
            "keyword_density": keyword_density,
            "coverage_ratio": coverage_ratio
        }

        coherence_score = None
        if len(sentences_raw) >= 2:
            lexical_similarities: List[float] = []
            for idx in range(len(sentence_sets)):
                base = sentence_sets[idx]
                for jdx in range(idx + 1, len(sentence_sets)):
                    compare = sentence_sets[jdx]
                    union = base | compare
                    if not union:
                        continue
                    intersection = base & compare
                    lexical_similarities.append(len(intersection) / len(union))

            vector_similarities: List[float] = []
            if SKLEARN_AVAILABLE:
                try:
                    tfidf = TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        min_df=1
                    ).fit_transform(sentences_raw)
                    sim_matrix = cosine_similarity(tfidf)
                    triu_indices = np.triu_indices(sim_matrix.shape[0], k=1)
                    vector_similarities = sim_matrix[triu_indices].tolist()
                except Exception as exc:
                    logger.debug(f"TF-IDF coherence fallback failed: {exc}")

            all_scores = lexical_similarities + vector_similarities
            if all_scores:
                coherence_score = round(float(np.mean(all_scores)) * 100, 1)

        if coherence_score is None:
            baseline = 45.0
            coverage_boost = min(1.0, len(top_topics) / max(len(sentences_raw), 1)) if sentences_raw else 0.0
            coherence_score = round((baseline + coverage_boost * 55.0), 1)

        avg_length = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        length_std = float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0
        short_pct = (
            len([length for length in sentence_lengths if length <= 8]) / len(sentence_lengths) * 100
            if sentence_lengths else 0.0
        )
        long_pct = (
            len([length for length in sentence_lengths if length >= 25]) / len(sentence_lengths) * 100
            if sentence_lengths else 0.0
        )
        variety_penalty = min(40, abs(avg_length - 18) * 2) + min(25, length_std * 1.5)
        pacing_penalty = min(15, short_pct * 0.35) + min(15, long_pct * 0.4)
        balance_bonus = 6 if 10 <= avg_length <= 20 else 0
        sentence_score = round(max(28, min(94, 94 - variety_penalty - pacing_penalty + balance_bonus)))
        sentence_pattern_breakdown = {
            "average_length": round(avg_length, 1),
            "length_std": round(length_std, 1),
            "short_pct": round(short_pct, 1),
            "long_pct": round(long_pct, 1)
        }

        fallback_repetition_alerts = []
        total_first_words = sum(first_word_counts.values())
        if total_first_words:
            for word, count in first_word_counts.most_common():
                proportion = count / total_first_words
                if proportion > 0.34 and word not in BASIC_STOPWORDS:
                    fallback_repetition_alerts.append({
                        "pattern": word,
                        "count": int(count),
                        "example": first_word_context.get(word, "")
                    })

        compression_ratio = self._compute_compression_ratio(text)
        distinct_metrics = self._compute_distinct_metrics(all_tokens)
        repeated_ngram_pct = self._compute_repeated_ngram_pct(all_tokens, n=4)
        subject_stats = self._heuristic_subject_verb_pairs(sentences_raw)

        return {
            "top_topics": top_topics,
            "keyword_coverage": keyword_coverage,
            "keyword_details": keyword_details,
            "topic_coherence_score": coherence_score,
            "sentence_pattern_score": sentence_score,
            "sentence_pattern_breakdown": sentence_pattern_breakdown,
            "repetition_alerts": fallback_repetition_alerts,
            "compression_ratio": compression_ratio,
            "pos_compression_ratio": None,
            "distinct_1": distinct_metrics.get("distinct_1"),
            "distinct_2": distinct_metrics.get("distinct_2"),
            "repeated_ngram_pct": repeated_ngram_pct,
            "subject_verb_stats": subject_stats
        }

    def _build_advanced_text_metrics(
        self,
        text: str,
        vocabulary_metrics: Dict,
        structure_metrics: Dict,
        content_metrics: Dict
    ) -> Dict:
        """
        Build topic coherence, keyword coverage, and sentence pattern analytics backed by spaCy vectors when available.
        """
        topic_coherence_score: Optional[float] = None
        top_topics: List[str] = []
        sentences_raw = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count_hint = structure_metrics.get('total_sentences') or len(sentences_raw) or 1
        keyword_coverage = {
            "total_keywords": 0,
            "top_keywords": [],
            "keyword_density": 0.0,
            "coverage_ratio": 0.0
        }
        sentence_pattern_score: Optional[float] = None
        repetition_alerts: List[Dict] = []
        keyword_details: List[Dict] = []
        sentence_pattern_breakdown: Dict[str, float] = {}
        token_pattern = re.compile(r'\b\w+\b')
        tokens = token_pattern.findall(text.lower())
        compression_ratio = self._compute_compression_ratio(text)
        distinct_metrics = self._compute_distinct_metrics(tokens)
        repeated_ngram_pct = self._compute_repeated_ngram_pct(tokens, n=4)
        pos_compression_ratio: Optional[float] = None
        subject_stats = structure_metrics.get('subject_verb_stats', {}) or {}

        total_words = max(1, vocabulary_metrics.get('total_words', 0) or 1)

        try:
            doc = self.spacy_model(text) if self.spacy_model and text.strip() else None
        except Exception as e:
            logger.warning(f"spaCy processing failed for advanced text metrics: {e}")
            doc = None

        if doc:
            sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 3]
            doc_vector = doc.vector
            doc_norm = np.linalg.norm(doc_vector)
            similarities: List[float] = []

            if doc_norm > 0 and sentences:
                for sent in sentences:
                    sent_vector = sent.vector
                    sent_norm = np.linalg.norm(sent_vector)
                    if sent_norm > 0:
                        sim = float(np.dot(sent_vector, doc_vector) / (sent_norm * doc_norm))
                        sim = max(0.0, min(1.0, sim))
                        similarities.append(sim)
                if similarities:
                    topic_coherence_score = round(float(np.mean(similarities)) * 100, 1)

            pos_tokens = [token.pos_ for token in doc if not token.is_space]
            if pos_tokens:
                pos_sequence = " ".join(pos_tokens)
                pos_compression_ratio = self._compute_compression_ratio(pos_sequence)

            chunk_examples: Dict[str, str] = {}
            noun_chunks = []
            for chunk in doc.noun_chunks:
                lemma = chunk.lemma_.lower().strip()
                if not lemma:
                    continue
                noun_chunks.append(lemma)
                chunk_examples.setdefault(lemma, chunk.sent.text.strip())
            if noun_chunks:
                chunk_counts = Counter(noun_chunks)
                top_topics = [phrase for phrase, _ in chunk_counts.most_common(8)]
                total_keywords = sum(chunk_counts.values())
                keyword_density = round((total_keywords / total_words) * 100, 2)
                coverage_ratio = round(len(set(top_topics)) / max(1, sentence_count_hint), 3)
                keyword_coverage = {
                    "total_keywords": total_keywords,
                    "top_keywords": top_topics[:6],
                    "keyword_density": keyword_density,
                    "coverage_ratio": coverage_ratio
                }
                keyword_details = [
                    {
                        "word": phrase,
                        "count": int(chunk_counts[phrase]),
                        "example": chunk_examples.get(phrase, "")
                    }
                    for phrase, _ in chunk_counts.most_common(6)
                ]
            elif content_metrics.get('key_topics'):
                top_topics = content_metrics.get('key_topics', [])[:8]
                keyword_coverage = {
                    "total_keywords": len(top_topics),
                    "top_keywords": top_topics[:6],
                    "keyword_density": round(len(top_topics) / total_words * 100, 2),
                    "coverage_ratio": round(len(top_topics) / max(1, sentence_count_hint), 3)
                }

            opener_pos: List[str] = []
            sentence_lengths: List[int] = []
            opener_examples: Dict[str, str] = {}
            for sent in sentences:
                tokens = [token for token in sent if not token.is_space]
                if not tokens:
                    continue
                first_token = tokens[0]
                opener_pos.append(first_token.pos_)
                sentence_lengths.append(len(tokens))
                key = first_token.pos_
                if key not in opener_examples:
                    opener_examples[key] = sent.text.strip()

            if opener_pos:
                pos_diversity = len(set(opener_pos)) / len(opener_pos)
                length_variance = np.var(sentence_lengths) if sentence_lengths else 0
                length_score = 1.0 - min(1.0, length_variance / 120.0)
                sentence_pattern_score = round((pos_diversity * 0.6 + length_score * 0.4) * 100, 1)
                sentence_pattern_breakdown = {
                    "average_length": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
                    "length_std": float(np.std(sentence_lengths)) if len(sentence_lengths) > 1 else 0.0,
                    "short_pct": round(len([l for l in sentence_lengths if l <= 8]) / len(sentence_lengths) * 100, 1) if sentence_lengths else 0.0,
                    "long_pct": round(len([l for l in sentence_lengths if l >= 25]) / len(sentence_lengths) * 100, 1) if sentence_lengths else 0.0
                }

                opener_counts = Counter(opener_pos)
                for pos_tag, count in opener_counts.most_common():
                    if count / len(opener_pos) > 0.35:
                        repetition_alerts.append({
                            "pattern": pos_tag,
                            "count": count,
                            "example": opener_examples.get(pos_tag)
                        })

            if not subject_stats:
                subject_stats = self._extract_subject_verb_pairs_doc(doc)

        if topic_coherence_score is None:
            coherence_raw = content_metrics.get('content_coherence')
            if isinstance(coherence_raw, (int, float)):
                topic_coherence_score = round(max(0.0, min(1.0, coherence_raw)) * 100, 1)
            else:
                topic_coherence_score = 0.0

        if not top_topics and content_metrics.get('key_topics'):
            top_topics = content_metrics.get('key_topics', [])[:6]
        if not top_topics and content_metrics.get('keyword_counts'):
            top_topics = list(content_metrics.get('keyword_counts', {}).keys())[:6]

        if keyword_coverage["total_keywords"] == 0 and content_metrics.get('key_topics'):
            key_topics = content_metrics.get('key_topics', [])
            keyword_coverage = {
                "total_keywords": len(key_topics),
                "top_keywords": key_topics[:6],
                "keyword_density": round(len(key_topics) / total_words * 100, 2),
                "coverage_ratio": round(len(key_topics) / max(1, sentence_count_hint), 3)
            }
            if not top_topics:
                top_topics = key_topics[:6]
        if keyword_coverage["total_keywords"] == 0 and content_metrics.get('keyword_counts'):
            keyword_counts = content_metrics.get('keyword_counts', {})
            total_keywords = sum(keyword_counts.values())
            top_keywords = list(keyword_counts.keys())[:6]
            keyword_coverage = {
                "total_keywords": total_keywords,
                "top_keywords": top_keywords,
                "keyword_density": round(total_keywords / total_words * 100, 2),
                "coverage_ratio": round(len(top_keywords) / max(1, sentence_count_hint), 3)
            }
            if not top_topics:
                top_topics = top_keywords

        if sentence_pattern_score is None:
            opener_diversity = structure_metrics.get('opener_diversity')
            if isinstance(opener_diversity, (int, float)):
                sentence_pattern_score = round(max(0.0, min(1.0, opener_diversity)) * 100, 1)
            else:
                sentence_pattern_score = 0.0

        if not repetition_alerts:
            weak_openers = structure_metrics.get('weak_openers', {}) or {}
            for pattern, count in weak_openers.items():
                if count > 0:
                    repetition_alerts.append({
                        "pattern": pattern,
                        "count": count,
                        "example": structure_metrics.get('opener_examples', {}).get(pattern)
                    })

        fallback_metrics = self._compute_simple_text_metrics(text, total_words)
        if not top_topics:
            top_topics = fallback_metrics["top_topics"]
        if keyword_coverage["total_keywords"] == 0 and fallback_metrics["keyword_coverage"]["total_keywords"] > 0:
            keyword_coverage = fallback_metrics["keyword_coverage"]
        if not keyword_details and fallback_metrics["keyword_details"]:
            keyword_details = fallback_metrics["keyword_details"]
        if topic_coherence_score is None or topic_coherence_score <= 0:
            topic_coherence_score = fallback_metrics["topic_coherence_score"]
        if sentence_pattern_score is None or sentence_pattern_score <= 0:
            sentence_pattern_score = fallback_metrics["sentence_pattern_score"]
        if not sentence_pattern_breakdown and fallback_metrics["sentence_pattern_breakdown"]:
            sentence_pattern_breakdown = fallback_metrics["sentence_pattern_breakdown"]
        if not repetition_alerts and fallback_metrics["repetition_alerts"]:
            repetition_alerts = fallback_metrics["repetition_alerts"]

        if pos_compression_ratio is None:
            pos_compression_ratio = fallback_metrics.get("pos_compression_ratio")
        if not subject_stats:
            subject_stats = fallback_metrics.get("subject_verb_stats", {})

        return {
            "topic_coherence_score": topic_coherence_score,
            "top_topics": top_topics,
            "keyword_coverage": keyword_coverage,
            "sentence_pattern_score": sentence_pattern_score,
            "repetition_alerts": repetition_alerts,
            "keyword_details": keyword_details,
            "sentence_pattern_breakdown": sentence_pattern_breakdown,
            "compression_ratio": compression_ratio,
            "pos_compression_ratio": pos_compression_ratio,
            "distinct_1": distinct_metrics.get("distinct_1"),
            "distinct_2": distinct_metrics.get("distinct_2"),
            "repeated_ngram_pct": repeated_ngram_pct,
            "subject_verb_diversity": subject_stats.get("diversity"),
            "subject_verb_pairs": subject_stats.get("top_pairs", [])
        }
    
    def analyze_vocabulary_with_openai(self, transcript: str) -> Optional[float]:
        """
        Use OpenAI to analyze vocabulary sophistication and quality
        
        Args:
            transcript: The speech transcript
            
        Returns:
            Vocabulary score from OpenAI or None if unavailable
        """
        if not self.openai_enhancer or not self.openai_enhancer.enabled:
            return None
        
        try:
            prompt = f"""Analyze this speech transcript and evaluate the vocabulary quality. Consider:
1. Word sophistication and complexity
2. Use of technical or professional terminology
3. Avoiding repetitive, simple words
4. Appropriate word choice for academic/business presentations
5. Overall lexical richness

Rate the vocabulary quality on a scale of 0-100 where:
- 0-40: Poor (repetitive, basic words, many "um/uh")
- 40-60: Average (some variation, mostly common words)
- 60-80: Good (varied vocabulary, some sophisticated words)
- 80-92: Excellent (rich vocabulary, sophisticated word choice)

Transcript: {transcript}

Respond ONLY with a number between 0-92 (no explanation)."""

            response = self.openai_enhancer.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a language assessment expert. Rate vocabulary quality and return only a number."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Ensure score is within realistic range
            score = max(30, min(92, score))
            
            logger.info(f"OpenAI vocabulary score: {score}")
            return score
            
        except Exception as e:
            logger.error(f"Error getting OpenAI vocabulary score: {e}")
            return None
    
    def analyze_text(self, text: str) -> Dict:
        """
        Complete text analysis
        
        Args:
            text: Input text
            
        Returns:
            Complete text analysis results
        """
        try:
            logger.info("Starting text analysis")
            
            if not text.strip():
                return {
                    'vocabulary_metrics': {},
                    'structure_metrics': {},
                    'content_metrics': {},
                    'vocabulary_score': 0,
                    'summary': 'No content to analyze.',
                    'analysis_successful': False,
                    'error': 'Empty text'
                }
            
            # Analyze vocabulary richness
            vocabulary_metrics = self.analyze_vocabulary_richness(text)
            
            # Analyze sentence structure
            structure_metrics = self.analyze_sentence_structure(text)
            
            # Analyze content quality
            content_metrics = self.analyze_content_quality(text)
            
            # Try OpenAI vocabulary scoring first (if available)
            openai_vocab_score = self.analyze_vocabulary_with_openai(text)
            
            # Always calculate local score too
            local_vocab_score = self.calculate_vocabulary_score(vocabulary_metrics, structure_metrics, content_metrics)
            
            # Combine OpenAI and local scores (weighted average)
            if openai_vocab_score is not None:
                # IMPROVED: Blend local (70%) and OpenAI (30%) - local is more comprehensive
                # VOCAB = 0.7 * VOC_INTERNAL + 0.3 * VOC_EXTERNAL
                vocabulary_score = (local_vocab_score * 0.7) + (openai_vocab_score * 0.3)
                
                # Apply small repetition penalty (avoid double punishment)
                type_token_ratio = vocabulary_metrics.get('type_token_ratio', 0.5)
                if type_token_ratio < 0.4:  # Low diversity suggests repetition
                    vocab_repetition_penalty = min(10, (0.4 - type_token_ratio) * 25)
                    vocabulary_score -= vocab_repetition_penalty
                    logger.info(f"Applied vocabulary repetition penalty: -{vocab_repetition_penalty:.1f} points (TTR: {type_token_ratio:.3f})")
                
                logger.info(f"Blended vocab score - Local: {local_vocab_score}, OpenAI: {openai_vocab_score}, Final: {vocabulary_score}")
            else:
                # Use only local if OpenAI unavailable
                vocabulary_score = local_vocab_score
                logger.info(f"Using local vocabulary score only: {vocabulary_score}")
            
            # Generate summary
            summary = self.generate_text_summary(text)

            advanced_text_metrics = self._build_advanced_text_metrics(
                text,
                vocabulary_metrics,
                structure_metrics,
                content_metrics
            )
            
            results = {
                'vocabulary_metrics': vocabulary_metrics,
                'structure_metrics': structure_metrics,
                'content_metrics': content_metrics,
                'vocabulary_score': vocabulary_score,
                'summary': summary,
                'advanced_text_metrics': advanced_text_metrics,
                'analysis_successful': True
            }
            
            logger.info("Text analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {
                'vocabulary_metrics': {},
                'structure_metrics': {},
                'content_metrics': {},
                'vocabulary_score': 0,
                'summary': 'Text analysis failed.',
                'analysis_successful': False,
                'error': str(e)
            }

# Global text analyzer instance
text_analyzer = TextAnalyzer()
