#!/usr/bin/env python

'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/midiparse.py
Purpose:    parse through midi and extract full grammar
            get corpus data to be used as input into lstm

Corpus data preparation inspired by public examples from the Keras Github.
    * [keras](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)

'''

from music21 import converter, note, stream
import copy


def getGrammars(midi_f):
    
    # Only one part for now
    midi = converter.parse(midi_f)[0]

    # Clean up anything not a Note or a Rest
    midi_clean = copy.deepcopy(midi)
    midi_clean.removeByNotOfClass(['Note', 'Rest'])

    # Construct Full Grammar
    fullGrammar = ""

    for i, n in enumerate(midi_clean):

        duration = float(n.duration.quarterLength)
        offset = float(n.offset)

        # Check Element, Pitch Name, Pitch Octave ~ for now, only check if Rest or Note
        element_type = ' '
        pitch_name = ' '
        pitch_octave = ' '

        if isinstance(n, note.Rest):
            element_type = 'R'
            pitch_name = 'X'
            pitch_octave = 'X'
        elif isinstance(n, note.Note):
            element_type = 'N'
            pitch_name = n.pitch.name
            pitch_octave = n.pitch.octave

        note_info = "%s,%s%s,%.3f" % (element_type, pitch_name, pitch_octave, offset)

        fullGrammar += (note_info + ' ')

    return(fullGrammar.rstrip())

def interpretGrammar(grammar):
    
    split_grammar = grammar.split(',')
    element_type = split_grammar[0]
    pitch = split_grammar[1]
    offset = split_grammar[2]

    if element_type == 'R':
        # Rest
        return (offset, note.Rest())
    elif element_type == 'N':
        # Note
        return (offset, note.Note(pitch))

def getCorpusData(fullGrammar):

    corpus = fullGrammar.split(' ')
    values = sorted(list(set(corpus)))

    return(corpus, values)
