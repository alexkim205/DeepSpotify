#!/usr/bin/env python

'''
Author:     Alex Kim, Justin Salamon
Project:    DeepMelodies
File:       src/meloextract.py
Purpose:    extract a melody

Borrowed heavily from Justin Salamon's 
* [Melodia Jupyter Notebook](http://www.justinsalamon.com/news/melody-extraction-in-python-with-melodia)

'''

import urllib.request
import vamp
import librosa
import numpy as np
import matplotlib.pyplot as plt

def extractMelody(data_file, fs=44100, hop=128):
    
    # Load 30s sample and extract melody
    mp3_f = urllib.request.urlretrieve(data_file)[0]
    audio, sr = librosa.load(mp3_f, sr=44100, mono=True)
    
    '''
    Melodia Parameters
        minfqr:  minimum frequency in Hertz (default 55.0)
        maxfqr:  maximum frequency in Hertz (default 1760.0)
        voicing: voicing tolerance. Greater values will result in more pitch contours included 
                in the final melody. Smaller values will result in less pitch contours 
                included in the final melody (default 0.2).
        minpeaksalience: (in Sonic Visualiser "Monophonic Noise Filter") is a hack to avoid 
                silence turning into junk contours when analyzing monophonic recordings (e.g. 
                solo voice with no accompaniment). Generally you want to leave this untouched 
                (default 0.0).
    '''
    params = {"minfqr": 55.0, "maxfqr": 1760.0, "voicing": 0.2, "minpeaksalience": 0.0}

    data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)
    #hop = data['vector'][0]
    melody = data['vector'][1]
    
    # the first timestamp is always 8 * hop
    hop = hop/fs
    first_timestamp = 8 * hop #= 8 * hop = 0.023219954648526078
    # Generate corresponding timestamp array
    timestamps = first_timestamp + np.arange(len(melody)) * hop
    
    '''
    Plot Extracted Melody
        Melodia returns unvoiced (=no melody) sections as negative values. A clearer option is to 
        get rid of the negative values before plotting. Finally, you might want to plot the pitch 
        sequence in cents rather than in Hz. This especially makes sense if you are comparing two 
        or more pitch sequences to each other (e.g. comparing an estimate against a reference).

    melody_cents = 1200*np.log2(melody/55.0)
    melody_cents[melody<=0] = None
    plt.figure(figsize=(18,6))
    plt.plot(timestamps, melody_cents)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (cents relative to 55 Hz)')
    plt.show()
    '''
    
    return(timestamps, melody, mp3_f)