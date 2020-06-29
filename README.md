# cs608proj
CS608 Project: EchoNest

Folder Structure:\
.\
+-- data\
|   +-- train_triplets.txt\
|   +-- unique_tracks.txt\
+-- db\
|   +-- lastfm_similars.db\
|   +-- lastfm_tags.db\
|   +-- mxm_dataset.db\
|   +-- track_metadata.db\
|   +-- artist_similarity.db\
|   +-- artist_term.db\
+-- notebooks\
+-- .gitignore

Data:\
Triplets: http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip\
Unique Tracks: http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt

DB:\
Track Metadata: http://millionsongdataset.com/sites/default/files/AdditionalFiles/track_metadata.db\
Artist Similarity: http://www.ee.columbia.edu/~thierry/artist_similarity.db\
Artist Term: http://www.ee.columbia.edu/~thierry/artist_term.db\
LastFM Tags: http://millionsongdataset.com/sites/default/files/lastfm/lastfm_tags.db\
LastFM Similarity: http://millionsongdataset.com/sites/default/files/lastfm/lastfm_similars.db\
MSM Dataset (lyrics): http://millionsongdataset.com/sites/default/files/AdditionalFiles/mxm_dataset.db

TODO:\
1) Notebook to explore dataset linking all DBs to Triplets. Sizing, EDA\
2) Query Spotify API for song features and analysis:\
+-- Search for song to get spotify song ID to use in subsequent query: https://developer.spotify.com/documentation/web-api/reference-beta/#endpoint-search\
+-- Audio Features: https://developer.spotify.com/documentation/web-api/reference-beta/#endpoint-get-audio-features\
+-- Audio Analysis: https://developer.spotify.com/documentation/web-api/reference-beta/#endpoint-get-audio-analysis