import os
import spotipy
import credentials
from spotipy.oauth2 import SpotifyClientCredentials
import urllib
import json 
import pandas as pd
import time


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
			'release_date': track['album']['release_date'],
			"uri": track['uri']
		}
	return False


def get_spotify_uri(sp, index, row):
	query = row.title + " " + row.artist_name
	record = spotify_search(sp, query)
	if record == False:
		query = row.title + " " + row.release
		record = spotify_search(sp, query)
		if record == False:
			record = {"query": query, "title": "", "artist": "", "album": "", "release_date": "", "uri": ""}
	record['track_id'] = row.track_id
	if index%10000==0:
		print(index)

	with open('spotify_ids.csv', 'a', encoding="utf-8") as textfile:
		record_feature = ["query", "title", "artist", "album", "release_date", "uri", "track_id"]
		write_string = "|".join([record.get(feature) for feature in record_feature]) + "\n"
		textfile.write(write_string)
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
							'liveness', 'valence', 'tempo', 'time_signature'
	]
	records = []
	bad_records = 0
	for idx, rec in enumerate(audio_features):
		try:
			records.append({feature: rec.get(feature) for feature in audio_features_features})
		except AttributeError:
			record = {
				'uri':uris[idx], 'danceability':-1, 'energy':-1, 'key':-1, 'loudness':-1, 'mode':-1, 
							'speechiness':-1, 'acousticness':-1, 'instrumentalness':-1, 
							'liveness':-1, 'valence':-1, 'tempo':-1, 'time_signature':-1
			}
			records.append(record)
			bad_records += 1
	if bad_records>0:
		print("num bads:", bad_records)

	df = pd.DataFrame(records)
	filename = "./spotify_audio_features.csv"
	if os.path.exists(filename):
		df.to_csv(filename, mode='a', header=False, index=False)
	else:
		df.to_csv(filename, mode='a', header=True, index=False)
	time.sleep(0.2)
	return records