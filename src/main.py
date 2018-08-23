#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/main.py
Purpose:    main script

'''

from __future__ import print_function
import logging, argparse
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from keras.models import load_model

from spotify import authenticateSpotify, getSpotifyData
from model_data import getModelData, KerasBatchGenerator
from lstm import createModel
from generate import generate

def main(run_opt):

    # Define model parameters
    n_epochs = 80
    n_songs = 6 # >= 3 ; train:validate:test = (n-2):1:1; last song will be test
    n_steps = 5 # number of timesteps in memory
    batch_size = 1
    skip_step = 1
    hidden_size = 128
    use_dropout = True

    # Define generate parameters
    n_predict = 10

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

    ## An integer: 1 to analyze audio, 2 to train, 3 to generate, 4 to do everything.
    
    # Analyze audio and write to files - Get training, validating, and testing data
    if (run_opt in [1, 4]): logging.info("Preparing training, validating, and testing data...")
    list_of_train_data, valid_data, test_data, values, val_indices, indices_val = \
        getModelData(sp, songs, media_output_dir, fs, hop, run_opt)

    # Train model
    if run_opt in [2, 4]:
        
        # Build model
        logging.info("Building model...")
        model, checkpointer = createModel(len(values), n_steps, hidden_size, use_dropout, model_output_dir)
        # print(model.summary())

        # Prepare batch generators and train
        logging.info("Prepare batch generators...")
        valid_data_generator = KerasBatchGenerator(valid_data, val_indices, n_steps, batch_size, len(values), skip_step=skip_step)
        #test_data_generator = KerasBatchGenerator(valid_data, n_steps, batch_size(valid_data), len(values), skip_step=skip_step)

        for i, train_data in enumerate(list_of_train_data):
            # Generate train data
            train_data_generator = KerasBatchGenerator(train_data, val_indices, n_steps, batch_size, len(values), skip_step=skip_step)
            # Train model
            print("len(train_data) - %d" % len(train_data))
            print("batch_size - %d" % batch_size)
            print("n_steps - %d" % n_steps)
            print("steps - %d" % (len(train_data)//(batch_size*n_steps)))
            model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*n_steps), n_epochs,
                                validation_data=valid_data_generator.generate(),
                                validation_steps=len(valid_data)//(batch_size*n_steps), callbacks=[checkpointer])

    # Generate new audio
    if run_opt in [3, 4]:

        model = load_model("%s/model-%d.hdf5" % (model_output_dir, n_epochs))
        dummy_iters = n_epochs
        example_test_generator = KerasBatchGenerator(test_data, val_indices, n_steps, batch_size, len(values), skip_step=skip_step)

        predicted_grammars = generate(model, test_data, example_test_generator, indices_val, n_steps, n_predict, dummy_iters)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('run_opt', type=int, default=4, help='An integer: 1 to analyze audio, 2 to train, 3 to generate, 4 to do everything.')
    args = parser.parse_args()

    main(args.run_opt)