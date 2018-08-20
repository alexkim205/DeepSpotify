#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/spotify.py
Purpose:    get preview urls of songs given Spotify URI

'''

import configparser
import sys, os, logging
import spotipy
import spotipy.oauth2 as oauth2


def authenticateSpotify(keyfile):
    
    # Parse for Spotify API Tokens
    config = configparser.ConfigParser()
    config.read(keyfile)
    print(keyfile)
    client_id = config.get('SPOTIFY', 'CLIENT_ID')
    client_secret = config.get('SPOTIFY', 'CLIENT_SECRET')

    # Authenticate
    auth = oauth2.SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    token = auth.get_access_token()
    sp = spotipy.Spotify(auth=token)

    return(sp)


def getSpotifyData(sp, uri):

    # Extract info from URI and request
    username = uri.split(':')[2]
    playlist_id = uri.split(':')[4]
    results = sp.user_playlist(username, playlist_id)
    songs = results['tracks']['items']

    return(songs)

