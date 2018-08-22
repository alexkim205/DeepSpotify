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
from model_data import getModelData, KerasBatchGenerator
from lstm import createModel

def main():

    # Define model parameters
    n_epochs = 50
    n_songs = 3 # >= 3 ; train:validate:test = (n-2):1:1; last song will be test
    n_steps = 30 # number of timesteps in memory
    batch_size = 2
    skip_step = 3
    hidden_size = 128
    use_dropout = True

    # Define analysis parameters
    fs = 44100
    hop = 128 

    media_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/melodia"
    model_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/model"
    key_f = "/Users/alexkim/Dropbox/Developer/ML/music/keys.cfg"
    spot_uri = "spotify:user:jeraldpgambino:playlist:6RzO3F2uNKdbRif0Fxo8Fn"

    # Get Spotify songs
    logging.info("Getting songs...")
    sp = authenticateSpotify(key_f)
    songs = getSpotifyData(sp, spot_uri, n_songs)

    # Get training, validating, and testing data
    logging.info("Preparing training, validating, and testing data...")
    list_of_train_data, valid_data, test_data, values, val_indices, indices_val = \
        getModelData(sp, songs, media_output_dir, fs, hop)

    # Build model
    model, checkpointer = createModel(len(values), n_steps, hidden_size, use_dropout, model_output_dir)
    print(model.summary())

    print(valid_data)
    print(len(valid_data))

    # Prepare batch generators and train
    logging.info("Prepare batch generators...")
    valid_data_generator = KerasBatchGenerator(valid_data, n_steps, batch_size, len(values), skip_step=skip_step)
    #test_data_generator = KerasBatchGenerator(valid_data, n_steps, batch_size(valid_data), len(values), skip_step=skip_step)

    for i, train_data in enumerate(list_of_train_data):
        # Generate train data
        train_data_generator = KerasBatchGenerator(train_data, n_steps, batch_size, len(values), skip_step=skip_step)
        # Train model
        model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*n_steps), n_epochs,
                            validation_data=valid_data_generator.generate(),
                            validation_steps=len(valid_data)//(batch_size*n_steps), callbacks=[checkpointer])
    

if __name__ == "__main__":
    
    main()