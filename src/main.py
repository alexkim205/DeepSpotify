#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/main.py
Purpose:    main script

'''

from __future__ import print_function
import logging, argparse, os
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
from keras.models import load_model
from music21 import stream, tempo, midi

from spotify import authenticateSpotify, getSpotifyData
from model_data import getModelData, KerasBatchGenerator
from lstm import createModel
from generate import generate
from midiparse import interpretGrammar
from melosynth import midiwrite

def main(run_opt):

    # Define model parameters
    n_epochs = 100
    n_songs = 6 # >= 3 ; train:validate:test = (n-2):1:1; last song will be test
    n_steps = 5 # number of timesteps in memory
    batch_size = 2
    skip_step = 1
    hidden_size = 128
    use_dropout = True

    # Define generate parameters
    # TODO - n_batch_prime = 1 # number of batches in test data to begin predicting with
    n_predict = 200

    # Define analysis parameters
    fs = 44100
    hop = 128 

    media_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/melodia"
    model_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/model"
    newsynth_output_dir = "/Users/alexkim/Dropbox/Developer/ML/music/data/new_synth"
    newsynth_f = "yellow_mellow.mid"
    key_f = "/Users/alexkim/Dropbox/Developer/ML/music/keys.cfg"
    spot_uri = "spotify:user:alezabeth1997:playlist:0grIqJ1svhAl7Lv9CIcKb6"

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
            logging.info("Training model[%d/%d]..." % (i, len(list_of_train_data)))
            model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*n_steps), n_epochs,
                                validation_data=valid_data_generator.generate(),
                                validation_steps=len(valid_data)//(batch_size*n_steps), callbacks=[checkpointer])

    # Generate new audio
    if run_opt in [3, 4]:

        # Load the latest model; TODO load the *best* model
        logging.info("Loading best model...")
        model = load_model("%s/model-%d.hdf5" % (model_output_dir, n_epochs))
        dummy_iters = n_epochs
        example_test_generator = KerasBatchGenerator(test_data, val_indices, n_steps, batch_size, len(values), skip_step=skip_step)

        # Obtain true grammars and predicted grammars
        logging.info("Predicting grammars...")
        true_grammars, predicted_grammars = generate(model, test_data, example_test_generator, indices_val, \
            n_steps, n_predict, dummy_iters)
        
        # Get bpm for the one test song
        test_bpm = sp.audio_analysis(songs[-1]['track']['uri'].split(':')[2])['track']['tempo']
        
        # print(true_grammars)
        # print(predicted_grammars)

        # Set up audio stream
        logging.info("Writing generated melody to MIDI...")
        true_notes = interpretGrammar(true_grammars)
        predicted_notes = interpretGrammar(predicted_grammars)
        
        newsynth_fo = os.path.join(newsynth_output_dir, newsynth_f)
        midiwrite(newsynth_fo, predicted_notes, test_bpm)


if __name__ == "__main__":
    
    arg_help = "An integer: 1 to analyze audio, 2 to train, 3 to generate, 4 to do everything."
    parser = argparse.ArgumentParser()
    parser.add_argument('run_opt', type=int, default=4, help=arg_help)
    args = parser.parse_args()

    main(args.run_opt)