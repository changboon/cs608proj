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


def spotify_search(sp, query):
	results = sp.search(q=query, limit=1)

	for idx, track in enumerate(results['tracks']['items']):
		return {
			"query": query, 
			"title": track['name'], 
			"artist": track['artists'][0]['name'],
			"album": track['album']['name'],
			"uri": track['uri']
		}
	return False


def get_spotify_uri(sp, row):
	query = row.title + " " + row.artist_name
	record = spotify_search(sp, query)
	if record == False:
		query = row.title + " " + row.release
		record = spotify_search(sp, query)
		if record == False:
			record = {"query": query, "title": "", "artist": "", "album": "", "uri": ""}
	record['track_id'] = row.track_id
	return record
		

def get_audio_analysis(sp, uri):
    analysis = sp.audio_analysis(uri)
    audio_analysis_features = ['uri', 'duration', 'loudness', 'tempo', 'tempo_confidence',
                               'time_signature', 'time_signature_confidence', 'key',
                                'key_confidence', 'mode', 'mode_confidence'
                              ]
    return {feature: analysis['track'].get(feature) for feature in audio_analysis_features}


def get_audio_features(sp, uris): 
	if len(uris) > 100:
		raise Exception("list's max size is 100")
    audio_features = sp.audio_features(uris) 
    audio_features_features = ['uri', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                               'speechiness', 'acousticness', 'instrumentalness', 
                               'liveness', 'valence', 'tempo'
                              ]
    records = []
    for rec in audio_features:
        records.append({feature: rec.get(feature) for feature in audio_features_features})
    return records