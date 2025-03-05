# hallucination_check.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Union, List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

class HallucinationChecker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the hallucination checker with a specific model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def _get_key_terms(self, text: str) -> set:
        """Extract key terms from text (excluding stop words)."""
        tokens = word_tokenize(text.lower())
        return set(word for word in tokens if word not in self.stop_words)

    def _calculate_term_overlap(self, actual: str, response: str) -> float:
        """Calculate the overlap of key terms between actual and response."""
        actual_terms = self._get_key_terms(actual)
        response_terms = self._get_key_terms(response)
        
        if not actual_terms or not response_terms:
            return 0.0
            
        overlap = len(actual_terms.intersection(response_terms))
        union = len(actual_terms.union(response_terms))
        return 1 - (overlap / union)

    def _calculate_semantic_distance(self, actual_embedding, response_embedding) -> float:
        """Calculate semantic distance using cosine similarity."""
        similarity = cosine_similarity(actual_embedding, response_embedding)[0][0]
        # Convert similarity to distance (0-1 range)
        return (1 - similarity) / 2

    def process_item(self, item: Union[str, List, Dict]) -> str:
        """Convert various input types to string format."""
        if isinstance(item, str):
            return item
        elif isinstance(item, (list, tuple)):
            return " ".join(str(x) for x in item)
        elif isinstance(item, dict):
            return " ".join(f"{k}: {v}" for k, v in item.items())
        return str(item)

    def check_hallucination(self, 
                          actual_data: Union[str, List[str], Dict], 
                          response_data: Union[str, List[str], Dict]) -> float:
        """
        Check the hallucination score between actual data and response.
        
        Args:
            actual_data: Ground truth data
            response_data: Response to check
            
        Returns:
            float: Hallucination score between 0 and 1
                  0 = No hallucination (high similarity)
                  1 = Complete hallucination (low similarity)
        """
        # Process inputs to strings
        actual_text = self.process_item(actual_data)
        response_text = self.process_item(response_data)
        
        # Get embeddings
        actual_embedding = self.model.encode(actual_text).reshape(1, -1)
        response_embedding = self.model.encode(response_text).reshape(1, -1)
        
        # Calculate semantic distance
        semantic_distance = self._calculate_semantic_distance(actual_embedding, response_embedding)
        
        # Calculate term overlap
        term_overlap = self._calculate_term_overlap(actual_text, response_text)
        
        # Combine scores with weights
        # Give more weight to term overlap for better sensitivity
        final_score = (0.4 * semantic_distance + 0.6 * term_overlap)
        
        # Apply exponential scaling to push subtle differences apart
        scaled_score = 1 - (1 - final_score) ** 2
        if scaled_score > 0.65:
            return 'Hallucinated', round(scaled_score,2)
        return 'Factual', round(scaled_score,2)
