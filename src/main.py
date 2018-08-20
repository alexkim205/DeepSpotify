#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/main.py
Purpose:    main script

'''

import logging

from spotify import authenticateSpotify, getSpotifyData
from melosynth import createMelody
from meloextract import extractMelody
from midiparse import getGrammars, getCorpusData

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def main():

    # define analysis parameters
    fs = 44100
    hop = 128

    media_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/melodia"
    key_f = "/Users/alexkim/Dropbox/Developer/ML/music/keys.cfg"
    spot_uri = "spotify:user:myplay.com:playlist:2f6tXtN0XesjONxicAzMIw"

    sp = authenticateSpotify(key_f)
    songs = getSpotifyData(sp, spot_uri)

    for song in songs[:2]:
        
        # Get name, uri, preview_url
        track = song['track']
        name = ''.join(e for e in track['name'] if e.isalnum())
        track_uri = track['uri'].split(':')[2]
        preview_url = track['preview_url']

        logging.info("Analyzing Spotify audio...")
        analysis = sp.audio_analysis(track_uri)
        bpm = analysis['track']['tempo']
        
        # If preview url is unavailable, skip song
        if (preview_url == None): continue
        
        # Extract and Write Melodies
        timestamps, melodyfreqs, orig_url = extractMelody(preview_url, fs, hop)
        csv_f, wav_f, wav_mix_f, wav_orig_f, midi_f = createMelody(timestamps, melodyfreqs, bpm, orig_url, media_output_dir, name)
        
        # Parse Midi to get Grammars
        logging.info("Extracting MIDI grammars...")
        grammar = getGrammars(midi_f)
        corpus, values, val_indices, indices_val = getCorpusData(grammar)
        print(values)

if __name__ == "__main__":
    
    main()