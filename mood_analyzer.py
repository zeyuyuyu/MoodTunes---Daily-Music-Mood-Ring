import pandas as pd
from datetime import datetime
from textblob import TextBlob
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

class MoodAnalyzer:
    def __init__(self, spotify_client_id=None, spotify_client_secret=None):
        self.mood_history = []
        self.spotify = None
        if spotify_client_id and spotify_client_secret:
            self.setup_spotify(spotify_client_id, spotify_client_secret)
    
    def setup_spotify(self, client_id, client_secret):
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.spotify = spotipy.Spotify(auth_manager=auth_manager)
    
    def analyze_mood(self, text):
        analysis = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        # Determine mood category
        if polarity > 0.5:
            mood = 'Energetic'
        elif polarity > 0:
            mood = 'Happy'
        elif polarity > -0.5:
            mood = 'Mellow'
        else:
            mood = 'Sad'
            
        mood_entry = {
            'date': datetime.now(),
            'text': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'mood': mood
        }
        
        self.mood_history.append(mood_entry)
        return mood_entry
    
    def get_song_recommendations(self, mood):
        if not self.spotify:
            raise Exception('Spotify credentials not configured')
            
        # Map moods to Spotify genres and attributes
        mood_mapping = {
            'Energetic': {
                'genres': ['edm', 'dance', 'power-pop'],
                'target_energy': 0.8,
                'target_valence': 0.8
            },
            'Happy': {
                'genres': ['pop', 'happy', 'feel-good'],
                'target_energy': 0.6,
                'target_valence': 0.7
            },
            'Mellow': {
                'genres': ['chill', 'ambient', 'indie'],
                'target_energy': 0.4,
                'target_valence': 0.5
            },
            'Sad': {
                'genres': ['sad', 'blues', 'rainy-day'],
                'target_energy': 0.3,
                'target_valence': 0.3
            }
        }
        
        mood_params = mood_mapping.get(mood, mood_mapping['Happy'])
        
        # Search for recommendations
        results = self.spotify.recommendations(
            seed_genres=mood_params['genres'],
            target_energy=mood_params['target_energy'],
            target_valence=mood_params['target_valence'],
            limit=5
        )
        
        recommendations = []
        for track in results['tracks']:
            recommendations.append({
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'url': track['external_urls']['spotify']
            })
            
        return recommendations
    
    def get_mood_history(self, days=7):
        df = pd.DataFrame(self.mood_history)
        df = df.set_index('date')
        return df.last(f'{days}D')
    
    def get_mood_stats(self):
        df = pd.DataFrame(self.mood_history)
        stats = {
            'total_entries': len(df),
            'mood_distribution': df['mood'].value_counts().to_dict(),
            'average_polarity': df['polarity'].mean(),
            'average_subjectivity': df['subjectivity'].mean()
        }
        return stats
