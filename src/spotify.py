import os
import spotipy
import credentials
from spotipy.oauth2 import SpotifyClientCredentials
import urllib

def load_and_auth_spotify(filepath):
	creds = credentials.load_spotify_creds(filepath)
	os.environ["SPOTIPY_CLIENT_ID"] = creds['client_id']
	os.environ["SPOTIPY_CLIENT_SECRET"] = creds['client_secret']

	sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(), requests_timeout=10)

	return sp


def get_spotify_uri(sp, query):
    results = sp.search(q=query, limit=1)
    record = {"query": query, "title": "", "artist": "", "album": "", "uri": ""}
    for idx, track in enumerate(results['tracks']['items']):
        return {
        "query": query, 
        "title": track['name'], 
        "artist": track['artists'][0]['name'],
        "album": track['album']['name'],
        "uri": track['uri']
        }
    return record


def get_audio_analysis(sp, uri):
    analysis = sp.audio_analysis(uri)
    audio_analysis_features = ['uri', 'duration', 'loudness', 'tempo', 'tempo_confidence',
                               'time_signature', 'time_signature_confidence', 'key',
                                'key_confidence', 'mode', 'mode_confidence'
                              ]
    return {feature: analysis['track'].get(feature) for feature in audio_analysis_features}


def get_audio_features(sp, uris): 
    audio_features = sp.audio_features(uris) 
    audio_features_features = ['uri', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                               'speechiness', 'acousticness', 'instrumentalness', 
                               'liveness', 'valence', 'tempo'
                              ]
    records = []
    for rec in audio_features:
        records.append({feature: rec.get(feature) for feature in audio_features_features})
    return records