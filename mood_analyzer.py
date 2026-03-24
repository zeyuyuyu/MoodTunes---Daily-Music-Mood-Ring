# MoodTunes Music Mood Analyzer
import spotipy
from textblob import TextBlob
from typing import Dict, List, Tuple

class MoodAnalyzer:
    def __init__(self, spotify_client: spotipy.Spotify):
        self.spotify = spotify_client
        self.mood_categories = {
            'happy': {'valence': (0.7, 1.0), 'energy': (0.6, 1.0)},
            'sad': {'valence': (0.0, 0.3), 'energy': (0.0, 0.4)},
            'relaxed': {'valence': (0.4, 0.7), 'energy': (0.0, 0.4)},
            'energetic': {'valence': (0.5, 1.0), 'energy': (0.7, 1.0)}
        }

    def analyze_text_mood(self, text: str) -> Tuple[str, float]:
        """Analyze text sentiment and map it to a musical mood category."""
        analysis = TextBlob(text)
        sentiment_score = analysis.sentiment.polarity
        
        # Map sentiment score to mood category
        if sentiment_score > 0.5:
            return ('happy', sentiment_score)
        elif sentiment_score < -0.3:
            return ('sad', sentiment_score)
        elif -0.3 <= sentiment_score <= 0.2:
            return ('relaxed', sentiment_score)
        else:
            return ('energetic', sentiment_score)

    def get_mood_recommendations(self, mood: str, limit: int = 5) -> List[Dict]:
        """Get song recommendations based on mood category."""
        if mood not in self.mood_categories:
            raise ValueError(f'Invalid mood category: {mood}')

        mood_params = self.mood_categories[mood]
        
        recommendations = self.spotify.recommendations(
            target_valence=(mood_params['valence'][0] + mood_params['valence'][1]) / 2,
            target_energy=(mood_params['energy'][0] + mood_params['energy'][1]) / 2,
            limit=limit
        )

        return [{
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'uri': track['uri'],
            'mood_category': mood
        } for track in recommendations['tracks']]

    def analyze_playlist_mood(self, playlist_id: str) -> Dict[str, float]:
        """Analyze the overall mood distribution of a playlist."""
        tracks = self.spotify.playlist_tracks(playlist_id)['items']
        mood_distribution = {mood: 0 for mood in self.mood_categories.keys()}
        
        for track in tracks:
            features = self.spotify.audio_features(track['track']['uri'])[0]
            if not features:
                continue
                
            valence = features['valence']
            energy = features['energy']
            
            for mood, params in self.mood_categories.items():
                if (params['valence'][0] <= valence <= params['valence'][1] and
                    params['energy'][0] <= energy <= params['energy'][1]):
                    mood_distribution[mood] += 1
                    break
        
        total = sum(mood_distribution.values())
        if total > 0:
            mood_distribution = {k: v/total for k, v in mood_distribution.items()}
            
        return mood_distribution