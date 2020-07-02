import json


# get credentials from spotify: https://developer.spotify.com/dashboard/
# cred file to be loaded should be json {"client_id": id, "client_secret":secret}
def load_spotify_creds(filepath):
	with open(filepath) as f:
		creds = json.load(f)
	return creds