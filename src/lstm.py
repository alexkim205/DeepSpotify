#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/lstm.py
Purpose:    create keras rnn model

'''

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from keras.callbacks import ModelCheckpoint

def createModel(n_values, n_steps, hidden_size, use_dropout, data_path):

    model = Sequential()
    model.add(Embedding(n_values, hidden_size, input_length=n_steps))
    # instead of model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(LSTM(128, return_sequences=True))
    if use_dropout: model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    if use_dropout: model.add(Dropout(0.2))
    model.add(Dense(n_values))
    model.add(Activation('softmax'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

    return(model, checkpointer)
