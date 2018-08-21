#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/lstm.py
Purpose:    create rnn here with keras

LSTM model inspired by public examples from the Keras Github.
    * [keras](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)

'''


import numpy as np
import random, sys
from keras.callbacks import LambdaCallback
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


def createModel(corpus, values, val_indices, indices_val, max_len, n_epochs=128):

    # no. of features; each unique value is a ~feature~
    n_values = len(values)

    # Build LSTM Model
    step = 3
    samples = [] # no. of samples in batch
    next_values = [] # 
    for i in range(0, len(corpus) - max_len, step):
        samples.append(corpus[i: i + max_len])
        next_values.append(corpus[i + max_len])
    print("Number of sequences: %d" % (len(samples)))

    # X = features
    # y = output
    x = np.zeros((len(samples), max_len, n_values), dtype=np.bool)
    y = np.zeros((len(samples), n_values), dtype=np.bool)
    for i, sample in enumerate(samples):
        for t, val in enumerate(sample):
            x[i, t, val_indices[val]] = 1
        y[i, val_indices[next_values[i]]] = 1
    
    # Build model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(max_len, n_values)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_values))
    model.add(Activation('softmax'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    # Fit model
    model.fit(x, y,
          batch_size=128,
          epochs=n_epochs)

    return(model)

