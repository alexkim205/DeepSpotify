#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/main.py
Purpose:    main script

'''

from __future__ import print_function
import logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

from spotify import authenticateSpotify, getSpotifyData
from model_data import getModelData
from lstm import createModel

def main():

    # Define model parameters
    n_epochs = 1
    n_songs = 10 # >= 3 ; train:validate:test = (n-2):1:1
    batch_size = 40 

    # Define analysis parameters
    fs = 44100
    hop = 128

    media_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/melodia"
    key_f = "/Users/alexkim/Dropbox/Developer/ML/music/keys.cfg"
    spot_uri = "spotify:user:jeraldpgambino:playlist:6RzO3F2uNKdbRif0Fxo8Fn"

    sp = authenticateSpotify(key_f)
    songs = getSpotifyData(sp, spot_uri, n_songs)
    list_of_train_data, valid_data, test_data, values, val_indices, indices_val = \
        getModelData(sp, songs, media_output_dir, fs, hop)

    print(list_of_train_data)
    
    model = None

    '''
    # Building Model
    logging.info("Building model...")
    m = createModel(corpus, values, val_indices, indices_val, batch_size, n_epochs)
    print(m.summary())
    '''
                

if __name__ == "__main__":
    
    main()