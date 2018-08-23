#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/model_data.py
Purpose:    get model data (train, validate, test data)

'''

import logging, os
import numpy as np
from keras.utils import to_categorical
from melosynth import createMelody
from meloextract import extractMelody
from midiparse import getGrammars, getCorpusData

def parseOneSong(sp, song, media_output_dir, fs=44100, hop=128, run_opt=4):

    # Get name, uri, preview_url
    track = song['track']
    name = ''.join(e for e in track['name'] if e.isalnum())
    track_uri = track['uri'].split(':')[2]
    preview_url = track['preview_url']

    if (run_opt in [1, 4]): logging.info("Analyzing Spotify audio...")
    analysis = sp.audio_analysis(track_uri)
    bpm = analysis['track']['tempo']

    midi_f = None
    if (run_opt in [1, 4]):
        # Extract and Write Melodies
        timestamps, melodyfreqs, orig_url = extractMelody(preview_url, fs, hop)
        csv_f, wav_f, wav_mix_f, wav_orig_f, midi_f = createMelody(timestamps, melodyfreqs, bpm, orig_url, media_output_dir, name)
    elif (run_opt in [2, 3]):
        # No need to write melodies to any files, just provide filename
        midi_f = os.path.join(media_output_dir, "synths", name + ".melo.midi")
        
    # Parse Midi to get Grammars
    logging.info("Extracting MIDI grammars...")
    grammar = getGrammars(midi_f)
    corpus, values = getCorpusData(grammar)
    
    return(corpus, values)

def hasPreview(song):
    return (song['track']['preview_url'] is not None)

def getModelData(sp, songs, media_output_dir, fs, hop, run_opt=4):
    '''
    split song data into three sets of corpuses: training, validation, and testing data
    Ratio is train:validate:test = (n-2):1:1
    '''

    # Filter only songs with available previews
    songs = [s for s in songs if hasPreview(s)]

    training_songs = songs[:-2]
    validating_song = songs[-2]
    testing_song = songs[-1]

    # Gather unique values from all sets of data
    values = []

    # Get corpus data for testing songs
    test_data, test_values = parseOneSong(sp, validating_song, media_output_dir, fs, hop, run_opt)
    values.extend(test_values)

    # Get corpus data for validating songs
    valid_data, valid_values = parseOneSong(sp, testing_song, media_output_dir, fs, hop, run_opt)
    values.extend(valid_values)

    # Get corpus data for training songs
    list_of_train_data = []

    for training_song in training_songs:
        train_corpus, train_values = parseOneSong(sp, training_song, media_output_dir, fs, hop, run_opt)
        list_of_train_data.append(train_corpus)
        values.extend(train_values)
    
    values = sorted(list(set(values)))
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))
    
    return (list_of_train_data, valid_data, test_data, values, val_indices, indices_val)


class KerasBatchGenerator(object):
    
    def __init__(self, data, val_indices, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data
        self.val_indices = val_indices
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :] = [self.val_indices[e] for e in self.data[self.current_idx:self.current_idx + self.num_steps]]
                temp_y = [self.val_indices[e] for e in self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
                self.current_idx += self.skip_step

            yield(x, y)

'''
    # Define model parameters
    n_epochs = 50
    n_songs = 3 # >= 3 ; train:validate:test = (n-2):1:1; last song will be test
    n_steps = 30 # number of timesteps in memory
    batch_size = 2
    skip_step = 3
    hidden_size = 128
    use_dropout = True
'''