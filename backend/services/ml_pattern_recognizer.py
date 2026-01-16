"""
ML Pattern Recognition for Weak Words
Uses machine learning to identify weak language patterns
"""
import logging
import numpy as np
from typing import Dict, List, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    import pickle
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")

class MLPatternRecognizer:
    """Machine learning-based pattern recognition for weak words"""
    
    def __init__(self):
        """Initialize ML pattern recognizer"""
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
        if ML_AVAILABLE:
            try:
                # Initialize simple model
                self.vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
                self.model = LogisticRegression(max_iter=1000, random_state=42)
                self.pipeline = Pipeline([
                    ('tfidf', self.vectorizer),
                    ('classifier', self.model)
                ])
                logger.info("ML pattern recognizer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize ML model: {e}")
                self.model = None
        else:
            logger.info("ML pattern recognition not available (scikit-learn not installed)")
    
    def extract_features(self, word: str, context: str, pos_tag: Optional[str] = None) -> Dict:
        """
        Extract features for ML model
        
        Args:
            word: Word to analyze
            context: Context sentence
            pos_tag: Part of speech tag
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Word-level features
        features['word_length'] = len(word)
        features['is_short'] = 1 if len(word) <= 3 else 0
        features['is_long'] = 1 if len(word) >= 8 else 0
        features['has_vowels'] = 1 if any(c in 'aeiou' for c in word.lower()) else 0
        
        # Context features
        context_lower = context.lower()
        features['word_position'] = context_lower.find(word.lower()) / max(len(context), 1)
        features['sentence_length'] = len(context.split())
        features['is_start'] = 1 if context_lower.startswith(word.lower()) else 0
        features['is_end'] = 1 if context_lower.endswith(word.lower()) else 0
        
        # POS features
        if pos_tag:
            features['is_adverb'] = 1 if pos_tag == 'ADV' else 0
            features['is_adjective'] = 1 if pos_tag == 'ADJ' else 0
            features['is_verb'] = 1 if pos_tag == 'VERB' else 0
        else:
            features['is_adverb'] = 0
            features['is_adjective'] = 0
            features['is_verb'] = 0
        
        return features
    
    def predict_weak_word(self, word: str, context: str, pos_tag: Optional[str] = None) -> Dict:
        """
        Predict if word is weak using ML model
        
        Args:
            word: Word to analyze
            context: Context sentence
            pos_tag: Part of speech tag
            
        Returns:
            Prediction results
        """
        try:
            if not self.model or not self.is_trained:
                # Fallback to rule-based
                return {
                    'is_weak': False,
                    'confidence': 0.5,
                    'method': 'rule_based_fallback'
                }
            
            # Create feature vector
            features = self.extract_features(word, context, pos_tag)
            feature_vector = np.array([list(features.values())])
            
            # Predict
            prediction = self.model.predict(feature_vector)[0]
            probability = self.model.predict_proba(feature_vector)[0]
            
            # Get confidence (probability of predicted class)
            confidence = float(max(probability))
            is_weak = bool(prediction == 1)
            
            return {
                'is_weak': is_weak,
                'confidence': confidence,
                'method': 'ml_prediction',
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return {
                'is_weak': False,
                'confidence': 0.5,
                'method': 'error_fallback'
            }
    
    def train_model(self, training_data: List[Dict]):
        """
        Train ML model on labeled data
        
        Args:
            training_data: List of dicts with 'word', 'context', 'label' (0=not weak, 1=weak)
        """
        if not ML_AVAILABLE or not self.model:
            logger.warning("ML not available, cannot train model")
            return
        
        try:
            # Prepare data
            X_text = [f"{item['word']} {item['context']}" for item in training_data]
            y = [item['label'] for item in training_data]
            
            # Train
            self.pipeline.fit(X_text, y)
            self.is_trained = True
            
            logger.info(f"ML model trained on {len(training_data)} examples")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            self.is_trained = False
    
    def get_weak_word_patterns(self, transcript: str) -> Dict:
        """
        Identify weak word patterns using ML
        
        Args:
            transcript: Full transcript
            
        Returns:
            Pattern analysis results
        """
        try:
            # Common weak words to check
            weak_candidates = ['just', 'really', 'very', 'quite', 'pretty', 'sort of', 'kind of',
                             'maybe', 'perhaps', 'probably', 'might', 'could', 'like', 'well',
                             'actually', 'basically', 'literally', 'thing', 'stuff', 'got', 'get']
            
            patterns = {
                'weak_words_detected': [],
                'pattern_types': Counter(),
                'total_weak_instances': 0
            }
            
            sentences = transcript.split('.')
            
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):
                    word_clean = word.lower().strip('.,!?;:()[]{}"\'')
                    
                    if word_clean in weak_candidates:
                        # Use ML to predict
                        prediction = self.predict_weak_word(word_clean, sentence)
                        
                        if prediction['is_weak'] and prediction['confidence'] > 0.6:
                            patterns['weak_words_detected'].append({
                                'word': word,
                                'context': sentence,
                                'confidence': prediction['confidence'],
                                'position': i
                            })
                            patterns['pattern_types'][word_clean] += 1
                            patterns['total_weak_instances'] += 1
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting weak word patterns: {e}")
            return {
                'weak_words_detected': [],
                'pattern_types': {},
                'total_weak_instances': 0
            }

# Global instance
ml_pattern_recognizer = MLPatternRecognizer()

