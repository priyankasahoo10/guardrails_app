# politeness_check.py
from nltk.sentiment import SentimentIntensityAnalyzer


def normalize_politeness_score(original_score):
    """
    Normalize the compound score to 0-1 range
    Negative scores indicate impolite, so we'll convert them
    """
    if original_score >= -0.5:
        return 0.0
    
    # Convert negative scores to positive scale
    # Map increasingly negative scores to higher unsafe scores
    # Use a scaling to ensure full range from 0 to 1
    normalized_score = (abs(original_score) - 0.5) / 0.5
    
    # Ensure the score is between 0 and 1
    return min(max(normalized_score, 0), 1.0)

# Updated politeness check with normalization
def politeness_check(response):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(response)
    
    # Determine politeness
    is_polite = sentiment['compound'] >= -0.5
    
    # Normalize the score
    normalized_score = normalize_politeness_score(sentiment['compound'])
    
    return ('Polite' if is_polite else 'Impolite', normalized_score)
