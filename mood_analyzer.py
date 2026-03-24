import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Tuple

class MoodAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
        # Define genre mappings based on emotional valence
        self.mood_genre_map = {
            'very_positive': ['Pop', 'Dance', 'Happy Rock', 'Disco'],
            'positive': ['Jazz', 'Soul', 'R&B', 'Folk'],
            'neutral': ['Classical', 'Ambient', 'Instrumental'],
            'negative': ['Blues', 'Alternative', 'Grunge'],
            'very_negative': ['Metal', 'Dark Ambient', 'Gothic']
        }

    def analyze_mood(self, text: str) -> Dict[str, float]:
        """Analyze the mood/sentiment of input text."""
        return self.sia.polarity_scores(text)

    def get_mood_category(self, sentiment_scores: Dict[str, float]) -> str:
        """Categorize the mood based on compound sentiment score."""
        compound = sentiment_scores['compound']
        
        if compound >= 0.5:
            return 'very_positive'
        elif 0.1 <= compound < 0.5:
            return 'positive'
        elif -0.1 <= compound < 0.1:
            return 'neutral'
        elif -0.5 <= compound < -0.1:
            return 'negative'
        else:
            return 'very_negative'

    def get_music_recommendations(self, text: str) -> Tuple[str, List[str]]:
        """Generate music genre recommendations based on mood analysis."""
        sentiment_scores = self.analyze_mood(text)
        mood_category = self.get_mood_category(sentiment_scores)
        recommended_genres = self.mood_genre_map[mood_category]
        
        return mood_category, recommended_genres

    def get_detailed_analysis(self, text: str) -> Dict:
        """Provide detailed mood analysis with music recommendations."""
        sentiment_scores = self.analyze_mood(text)
        mood_category = self.get_mood_category(sentiment_scores)
        recommended_genres = self.mood_genre_map[mood_category]
        
        return {
            'sentiment_scores': sentiment_scores,
            'mood_category': mood_category,
            'recommended_genres': recommended_genres,
            'input_text': text
        }

if __name__ == '__main__':
    # Example usage
    analyzer = MoodAnalyzer()
    sample_text = 'I feel absolutely amazing today!'
    
    analysis = analyzer.get_detailed_analysis(sample_text)
    print(f"Input: {analysis['input_text']}")
    print(f"Mood Category: {analysis['mood_category']}")
    print(f"Sentiment Scores: {analysis['sentiment_scores']}")
    print(f"Recommended Music Genres: {', '.join(analysis['recommended_genres'])}")