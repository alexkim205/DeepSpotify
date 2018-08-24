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

from music21 import converter, note, stream, midi
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

        note_info = "%s,%s%s,%.3f" % (element_type, pitch_name, pitch_octave, duration)

        fullGrammar += (note_info + ' ')

    return(fullGrammar.rstrip())


def interpretGrammar(grammars):
    
    curr_offset = 0
    notes = []

    for g in grammars:

        split_grammar = g.split(',')
        element_type = split_grammar[0]
        pitch = split_grammar[1]
        duration = float(split_grammar[2])

        if element_type == 'R':
            # Rest
            # Don't add anything to notes list
            None
            # element = midi.translate.noteToMidiEvents(note.Rest())
            # element.duration.quarterLength = float(duration)
        elif element_type == 'N':
            # Note
            n = note.Note(pitch)
            midi_pitch = midi.translate.noteToMidiEvents(n)[1].pitch
            notes.append((curr_offset, duration, midi_pitch))
        
        curr_offset += duration

    return (notes)


def getCorpusData(fullGrammar):

    corpus = fullGrammar.split(' ')
    values = sorted(list(set(corpus)))

    return(corpus, values)
