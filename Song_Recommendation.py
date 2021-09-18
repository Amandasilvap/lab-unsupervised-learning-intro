#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function    # (at top of module)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import cluster
from random import randint
from time import sleep
from spotipy.oauth2 import SpotifyClientCredentials
import json
import spotipy
import sys
import requests
import spotipy.util as util
from random import randint
from time import sleep
import getpass

client_id = getpass.getpass()
client_secret = getpass.getpass()
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))
playlist_spotify = pd.read_csv("/Users/amandamac/IronHackerDA/lab-web-scraping-single-page/playlist_spotify.csv")
hot_songs = pd.read_csv("/Users/amandamac/IronHackerDA/lab-web-scraping-single-page/hot_songs.csv")
playlist_spotify = playlist_spotify.drop("Unnamed: 0", axis=1)
playlist_spotify_model = playlist_spotify.select_dtypes(include=np.number)


scaler = StandardScaler().fit(playlist_spotify_model)
X_prep = scaler.transform(playlist_spotify_model)
pd.DataFrame(X_prep,columns=playlist_spotify_model.columns).head()
kmeans = KMeans(n_clusters=9, n_init = 10, random_state = 1234)
y_kmeans = kmeans.fit_predict(X_prep)
clusters = kmeans.predict(X_prep)
pd.Series(clusters).value_counts().sort_index()
clusters = pd.Series(clusters)
playlist_spotify_cluster = pd.concat([playlist_spotify, clusters], axis=1)
playlist_spotify_cluster= playlist_spotify_cluster.rename(columns={0: 'cluster'})
playlist_spotify_cluster.to_csv("playlist_spotify_cluster.csv")

def features(track, artist):
    query = track + " " + artist
    track_id = sp.search(q= query, type="track,artist", limit=10)
    if len(track_id["tracks"]["items"]) > 0:
        uri= track_id["tracks"]["items"][0]["uri"]
        features = sp.audio_features(uri)
        song_df = pd.DataFrame(features)
        song_df = song_df.select_dtypes(include=np.number)
        song_df = scaler.transform(song_df)
        clusters_new_song = kmeans.predict(song_df)
        clusters_new_song = pd.Series(clusters_new_song).values[0]
        return clusters_new_song
    
    return -1

print("Hi, let's play a song! Choose a music and an artist:")
sleep(randint(1,3))

song = input("Song Name: ")
artist= input("Artist name: ")

choice = hot_songs.loc[(hot_songs['song'].str.lower() == song.lower()) & (hot_songs['artist'].str.lower() == artist.lower())]

if len(choice) > 0:
    suggest = hot_songs.sample()
    print("Maybe you can like this another hot song:", suggest['song'], suggest['artist'])
    sleep(randint(1,3))
else:
    print("Not Found in Hot songs, searching in our recomendations")
    choice2 = playlist_spotify_cluster.loc[(playlist_spotify_cluster['track_name'].str.lower() == song.lower()) & (playlist_spotify_cluster['artist'].str.lower() == artist.lower())]
    if len(choice2) > 0:
        suggest = playlist_spotify_cluster.loc[playlist_spotify_cluster["cluster"]==choice2["cluster"].values[0]].sample()
        print("Maybe you can like this music: ", suggest['track_name'].values[0],"- ", suggest['artist'].values[0])
        sleep(randint(1,3))
    else: 
        cluster_features = features(song, artist)
        if cluster_features >=0:
            suggest = playlist_spotify_cluster.loc[playlist_spotify_cluster["cluster"]==cluster_features].sample()
            print("Maybe you can like this song: ", suggest['track_name'].values[0],"- ", suggest['artist'].values[0])
            sleep(randint(1,3))
        else:
            print("I can't find this song. Try another music")


