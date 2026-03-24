import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Tuple

class MoodAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        
        # Define genre mappings based on mood and intensity
        self.mood_genre_map = {
            'very_positive': ['Pop', 'Dance', 'Electronic'],
            'positive': ['Indie Pop', 'Folk', 'Jazz'],
            'neutral': ['Classical', 'Ambient', 'Lo-fi'],
            'negative': ['Blues', 'Alternative', 'Indie Rock'],
            'very_negative': ['Metal', 'Punk', 'Grunge']
        }

    def analyze_mood(self, text: str) -> Tuple[str, float, list]:
        """
        Analyzes text to determine mood, intensity, and suggested music genres
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple containing:
            - mood classification (str)
            - mood intensity score (float)
            - list of recommended genres (list)
        """
        scores = self.sia.polarity_scores(text)
        compound_score = scores['compound']
        
        # Determine mood category and intensity
        if compound_score >= 0.5:
            mood = 'very_positive'
            intensity = abs(compound_score)
        elif 0 < compound_score < 0.5:
            mood = 'positive'
            intensity = abs(compound_score)
        elif compound_score == 0:
            mood = 'neutral'
            intensity = 0.0
        elif -0.5 < compound_score < 0:
            mood = 'negative'
            intensity = abs(compound_score)
        else:
            mood = 'very_negative'
            intensity = abs(compound_score)
            
        recommended_genres = self.mood_genre_map[mood]
        
        return mood, intensity, recommended_genres

    def get_detailed_analysis(self, text: str) -> Dict:
        """
        Provides detailed mood analysis including raw sentiment scores
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dictionary containing detailed analysis
        """
        scores = self.sia.polarity_scores(text)
        mood, intensity, genres = self.analyze_mood(text)
        
        return {
            'mood': mood,
            'intensity': intensity,
            'recommended_genres': genres,
            'raw_scores': {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound']
            }
        }

if __name__ == '__main__':
    analyzer = MoodAnalyzer()
    
    # Example usage
    sample_text = 'I am feeling absolutely amazing today!'
    mood, intensity, genres = analyzer.analyze_mood(sample_text)
    print(f'Mood: {mood}')
    print(f'Intensity: {intensity:.2f}')
    print(f'Recommended Genres: {genres}')
    
    # Detailed analysis
    detailed = analyzer.get_detailed_analysis(sample_text)
    print('\nDetailed Analysis:')
    print(detailed)