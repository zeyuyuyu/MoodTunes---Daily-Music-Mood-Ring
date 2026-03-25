# MoodTunes Mood Analyzer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob

class MoodAnalyzer:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.mood_categories = ['happy', 'sad', 'energetic', 'calm', 'anxious']
        
    def analyze_text_sentiment(self, text):
        """Analyze text sentiment using TextBlob and return mood classification"""
        analysis = TextBlob(text)
        
        # Extract features
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Create feature vector
        features = np.array([[polarity, subjectivity]])
        
        # Map sentiment scores to mood categories
        if polarity > 0.5:
            return 'happy'
        elif polarity < -0.3:
            return 'sad'
        elif subjectivity > 0.8:
            return 'anxious'
        elif polarity > 0 and subjectivity < 0.4:
            return 'energetic'
        else:
            return 'calm'
    
    def get_mood_playlist(self, mood):
        """Return music recommendations based on mood"""
        playlists = {
            'happy': ['upbeat_pop', 'feel_good_hits', 'summer_vibes'],
            'sad': ['melancholic_ballads', 'acoustic_covers', 'rainy_day'],
            'energetic': ['workout_hits', 'dance_party', 'power_rock'],
            'calm': ['ambient_chill', 'peaceful_piano', 'meditation'],
            'anxious': ['calming_classical', 'nature_sounds', 'gentle_acoustic']
        }
        return playlists.get(mood, [])
    
    def analyze_mood_from_journal(self, journal_text):
        """Analyze mood from journal entry and return music recommendations"""
        mood = self.analyze_text_sentiment(journal_text)
        recommendations = self.get_mood_playlist(mood)
        
        return {
            'detected_mood': mood,
            'confidence_score': abs(TextBlob(journal_text).sentiment.polarity),
            'playlist_recommendations': recommendations
        }

    def get_mood_description(self, mood):
        """Return detailed description of detected mood"""
        descriptions = {
            'happy': 'You seem to be in high spirits! Your positivity is shining through.',
            'sad': 'You appear to be feeling down. Music can help lift your spirits.',
            'energetic': 'You have high energy levels and seem ready for action!',
            'calm': 'You are in a balanced and peaceful state of mind.',
            'anxious': 'You seem to have some worried thoughts. Calming music might help.'
        }
        return descriptions.get(mood, 'Unable to determine mood description')
