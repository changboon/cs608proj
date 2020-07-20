#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import crud as crud
import pandas as pd
import matplotlib

from sklearn.model_selection import train_test_split

import cornac
from cornac.data import Reader
from cornac.eval_methods import BaseMethod, RatioSplit
from cornac.models import BPR, WMF


# In[5]:


triplets = pd.read_csv('../data/train_triplets.txt', sep="\t", header=None)
triplets.columns=['user_id', 'song_id', 'play_count']


# In[6]:


MIN_PLAYCOUNT = 5
MAX_PLAYCOUNT = 150
MIN_RATING = 50
SEED = 2020
TEST_RATIO = 0.33

triplets = triplets[(triplets['play_count'] >= MIN_PLAYCOUNT) & (triplets['play_count'] <= MAX_PLAYCOUNT)]
grouped = triplets.groupby('song_id')
triplets = grouped.filter(lambda x: x["user_id"].nunique() >= MIN_RATING)
triplets['user_id_idx'] = triplets.groupby('user_id').ngroup()
triplets['song_id_idx'] = triplets.groupby('song_id').ngroup()


# In[7]:


triplets.shape


# In[4]:


triplets.head()


# In[5]:


triplets.describe()


# In[6]:


train, test = train_test_split(triplets, test_size=TEST_RATIO, shuffle=True, random_state=SEED)
train_triplets = pd.DataFrame(train, columns=["user_id", "song_id", "play_count", "user_id_idx", "song_id_idx"])
test_triplets = pd.DataFrame(test, columns=["user_id", "song_id", "play_count", "user_id_idx", "song_id_idx"])


# In[7]:


from pathlib import Path

output_dir = Path('../experiments/triplets')
output_dir.mkdir(parents=True, exist_ok=True)

train_triplets.to_csv(output_dir / "train.csv", index=False)
test_triplets.to_csv(output_dir / "test.csv", index=False)


# In[8]:


conn = crud.create_connection("../db/track_metadata.db")
tables = crud.get_tables(conn)

for table_name in tables.name.tolist():
    print(table_name)
    records = crud.get_records(conn, table_name)
    #print(records.head())


# In[9]:


tracks = records[['track_id', 'title', 'song_id']]
tracks.to_csv('./tracks.csv', index=False)
tracks.shape


# In[10]:


records_columns = ['song_id', 'artist_id', 'duration', 'artist_familiarity', 'artist_hotttnesss']
songs = records[records_columns].groupby(['song_id', 'artist_id']).mean().reset_index()
songs.shape


# In[11]:


n_users = triplets.user_id.nunique()
n_songs = triplets.song_id.nunique()


# In[12]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{}\n".format(
        rating,
        uid,
        sid + song_start_idx, 
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness
    )

output_dir = Path('../experiments/metadata')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 'duration', 'artist_familiarity', 'artist_hotttnesss']

train_df = (
    train_triplets
    .merge(songs.drop(columns=['artist_id']), on='song_id')
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs.drop(columns=['artist_id']), on='song_id')
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness))


# In[13]:


conn = crud.create_connection("../db/artist_term.db")
tables = crud.get_tables(conn)

for table_name in tables.name.tolist():
    print(table_name)
    records = crud.get_records(conn, table_name)
    print(records.shape)
    if table_name == "artist_term":
        break


# In[14]:


artist_term = records
artist_term['term_id'] = artist_term.groupby('term').ngroup()
n_terms = artist_term.term_id.max()
artist_term_lookup = artist_term[['artist_id', 'term_id']].drop_duplicates()
artist_term_lookup.to_csv("./artist_term_lookup.csv", index=False)
artist_term = artist_term[['artist_id', 'term_id']].groupby('artist_id').agg(list).reset_index()
artist_term.to_csv("./artist_term.csv", index=False)
artist_term.shape


# In[15]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    artist_term_start_idx = hotness_start_idx + 1
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {}\n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:1".format(t + artist_term_start_idx) for t in terms)
    )

output_dir = Path('../experiments/metadata_artist')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'term_id']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(artist_term, on="artist_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(artist_term, on="artist_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms))


# In[16]:


conn = crud.create_connection("../db/mxm_dataset.db")
tables = crud.get_tables(conn)

for table_name in tables.name.tolist():
    print(table_name)
    records = crud.get_records(conn, table_name)
    print(records.shape)


# In[17]:


lyrics = records 
lyrics['word_id'] = lyrics.groupby('word').ngroup()
n_lyrics = lyrics.word_id.max()
lyrics_word_lookup = lyrics[['word_id', 'word']].drop_duplicates()
lyrics_word_lookup.to_csv("./lyrics_word_lookup.csv", index=False)
lyrics = lyrics.merge(tracks[['track_id', 'song_id']], on='track_id')[['song_id', 'word_id', 'count']].drop_duplicates()
lyrics = lyrics.groupby('song_id')[['word_id', 'count']].apply(lambda g: g.values.tolist()).reset_index()
lyrics.columns = ['song_id', 'lyrics']
lyrics.to_csv("./lyrics.csv", index=False)
lyrics.shape


# In[18]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, lyric):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    lyric_start_idx = hotness_start_idx + 1
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {}\n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:{}".format(t[0] + lyric_start_idx, t[1]) for t in lyric)
    )

output_dir = Path('../experiments/metadata_lyrics')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'lyrics']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, lyric, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, lyric))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, lyric, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, lyric))


# In[19]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    artist_term_start_idx = hotness_start_idx + 1
    lyric_start_idx = artist_term_start_idx + n_terms
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {} {}\n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:1".format(t + artist_term_start_idx) for t in terms),
        " ".join("{}:{}".format(t[0] + lyric_start_idx, t[1]) for t in lyric)
    )

output_dir = Path('../experiments/metadata_artist_lyrics')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 'duration', 'artist_familiarity', 'artist_hotttnesss', 'term_id', 'lyrics']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .merge(artist_term, on="artist_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .merge(artist_term, on="artist_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, lyric, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, lyric, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric))


# In[20]:


spotify_id = pd.read_csv('./spotify_ids.csv', header=None, sep='|', error_bad_lines=False, warn_bad_lines=False, names=
                  ['query', 'spotify_title', 'spotify_artist', 'spotify_album', 'spotify_release', 'spotify_uri', 'track_id']
                  )
spotify_id = spotify_id.loc[~spotify_id.spotify_uri.isna()]
spotify_id = spotify_id.merge(tracks[['track_id', 'song_id']], on='track_id')[['spotify_uri', 'song_id', 'spotify_release']].drop_duplicates()
spotify_id.shape


# In[21]:


audio_features = pd.read_csv("./spotify_audio_features.csv")
audio_features = audio_features.add_prefix('spotify_')
print(audio_features.shape)
audio_features = audio_features.loc[audio_features.spotify_energy!=-1]
print(audio_features.shape)
audio_features.shape


# In[22]:


spotify = spotify_id.merge(audio_features, on="spotify_uri").drop(columns=['spotify_uri', 'spotify_release']).drop_duplicates()
spotify = spotify.groupby('song_id').mean().reset_index()
print(spotify.song_id.nunique())
print(spotify.shape)
spotify_cols = [col for col in spotify.columns if col != 'song_id']
spotify = pd.melt(spotify, id_vars=['song_id'], value_vars=spotify_cols)
spotify = spotify.groupby('song_id')[['variable', 'value']].apply(lambda g: g.values.tolist()).reset_index()
spotify.columns = ['song_id', 'spotify']
spotify.shape


# In[23]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric, spotify_audio):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    artist_term_start_idx = hotness_start_idx + 1
    lyric_start_idx = artist_term_start_idx + n_terms
    audio_start_idx = lyric_start_idx + n_lyrics
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {} {} {} \n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:1".format(t + artist_term_start_idx) for t in terms),
        " ".join("{}:{}".format(l[0] + lyric_start_idx, l[1]) for l in lyric),
        " ".join("{}:{}".format(spotify_cols.index(s[0]) + audio_start_idx, s[1]) for s in spotify_audio)
    )

output_dir = Path('../experiments/metadata_artist_lyrics_spotify')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 
                'duration', 'artist_familiarity', 'artist_hotttnesss', 
                'term_id', 'lyrics', 'spotify']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .merge(artist_term, on="artist_id")
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(lyrics, on="song_id")
    .merge(artist_term, on="artist_id")
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, lyric, spotify_audio, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric, spotify_audio))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, lyric, spotify_audio, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, lyric, spotify_audio))


# In[28]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, spotify_audio):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    audio_start_idx = hotness_start_idx + 1
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {}\n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:{}".format(spotify_cols.index(s[0]) + audio_start_idx, s[1]) for s in spotify_audio)
    )

output_dir = Path('../experiments/metadata_spotify')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 
                'duration', 'artist_familiarity', 'artist_hotttnesss', 
                'spotify']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, spotify_audio, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, spotify_audio))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, spotify_audio, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, spotify_audio))


# In[29]:


from pathlib import Path

def to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, spotify_audio):
  # order of features: user, movie, tags
    user_start_idx = 0
    song_start_idx = n_users
    duration_start_idx = song_start_idx + n_songs
    familiarity_start_idx = duration_start_idx + 1
    hotness_start_idx = familiarity_start_idx + 1
    artist_term_start_idx = hotness_start_idx + 1
    audio_start_idx = artist_term_start_idx + n_terms
    
    return "{} {}:1 {}:1 {}:{} {}:{} {}:{} {} {}\n".format(
        rating,
        uid,
        sid + song_start_idx,
        duration_start_idx, duration,
        familiarity_start_idx, familiarity,
        hotness_start_idx, hotness,
        " ".join("{}:1".format(t + artist_term_start_idx) for t in terms),
        " ".join("{}:{}".format(spotify_cols.index(s[0]) + audio_start_idx, s[1]) for s in spotify_audio)
    )

output_dir = Path('../experiments/metadata_artist_spotify')
output_dir.mkdir(parents=True, exist_ok=True)

meta_columns = ['play_count', 'user_id_idx', 'song_id_idx', 
                'duration', 'artist_familiarity', 'artist_hotttnesss', 
                'term_id', 'spotify']

train_df = (
    train_triplets
    .merge(songs, on='song_id')
    .merge(artist_term, on="artist_id")
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(train_df.shape)
test_df = (
    test_triplets
    .merge(songs, on='song_id')
    .merge(artist_term, on="artist_id")
    .merge(spotify, on="song_id")
    .drop(columns=['artist_id'])
)
print(test_df.shape)

# save training data to file
with open(output_dir / "train.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, spotify_audio, *_ in train_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, spotify_audio))

# save test data to file
with open(output_dir / "test.libfm", "w") as f:
    for rating, uid, sid, duration, familiarity, hotness, terms, spotify_audio, *_ in test_df[meta_columns].itertuples(index=False):
        f.write(to_fm_sparse_fmt(rating, uid, sid, duration, familiarity, hotness, terms, spotify_audio))

