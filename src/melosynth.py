#!/usr/bin/env python

'''
Author:     Alex Kim, Justin Salamon
Project:    MeloSynth / (used in) DeepMelodies
File:       src/melosynth.py
Purpose:    synthesize a melody

Borrowed heavily from combination of Justin Salamon's 
    * [Melosynth](https://github.com/justinsalamon/melosynth)
    * [audio_to_midi_melodia](https://github.com/justinsalamon/audio_to_midi_melodia)

'''

import os, wave, logging
import numpy as np
import scipy.io.wavfile
from scipy.signal import medfilt
from midiutil.MidiFile import MIDIFile
import librosa


def createMelody(_times, _freqs, _bpm, _orig_url, _outputfolder, _outputid):
    
    '''
    def melosynth(times, freqs, bpm, orig_url, outputfolder, outputid, 
                  nHarmonics=1, square=False, useneg=False,
                  fs=44100, hop=128, smooth=0.25, minduration=0.1)
    '''
    
    return(melosynth(_times, _freqs, _bpm, _orig_url, _outputfolder, _outputid, 
        nHarmonics=1, square=False, useneg=False, 
        fs=44100, hop=128, smooth=0.25, minduration=0.1))

def wavwrite(x, filename, fs=44100, N=16):
    '''
    Synthesize signal x into a wavefile on disk. The values of x must be in the
    range [-1,1].

    :parameters:
    - x : numpy.array
    Signal to synthesize.

    - filename: string
    Path of output wavfile.

    - fs : int
    Sampling frequency, by default 44100.

    - N : int
    Bit depth, by default 16.
    '''
    
    maxVol = 2**15-1.0 # maximum amplitude
    x = x * maxVol # scale x
    
    scipy.io.wavfile.write(filename, fs, np.asarray(x, dtype=np.int16))

def midiwrite(outfile, notes, tempo):

    track = 0
    time = 0
    midifile = MIDIFile(1) # Just one track for now
    channel = 0
    volume = 100
    program = 54 # Synth Voice
    
    # Add track name and tempo.
    midifile.addTrackName(track, time, "Synth Track")
    midifile.addTempo(track, time, tempo)
    midifile.addProgramChange(track, channel, time, program)

    for note in notes:
        onset = note[0] * (tempo/60.)
        duration = note[1] * (tempo/60.)
        # duration = 1
        pitch = note[2].astype(int)
        midifile.addNote(track, channel, pitch, onset, duration, volume)

    # And write it to disk.
    binfile = open(outfile, 'wb')
    midifile.writeFile(binfile)
    binfile.close()

def midi_to_notes(midi, fs=44100, hop=128, smooth=0.25, minduration=0.1):
    '''
    Load pitch sequence from  a txt/csv file and synthesize it into a .wav
    '''
    
    # smooth midi pitch sequence first
    if (smooth > 0):
        filter_duration = smooth  # in seconds
        filter_size = int(filter_duration * fs / float(hop))
        if filter_size % 2 == 0:
            filter_size += 1
        midi_filt = medfilt(midi, filter_size)
    else:
        midi_filt = midi
    
    
    notes = []
    p_prev = 0
    duration = 0
    onset = 0
    for n, p in enumerate(midi_filt):
        if p == p_prev:
            duration += 1
        else:
            # treat 0 as silence
            if p_prev > 0:
                # add note
                duration_sec = duration * hop / float(fs)
                # only add notes that are long enough
                if duration_sec >= minduration:
                    onset_sec = onset * hop / float(fs)
                    notes.append((onset_sec, duration_sec, p_prev))

            # start new note
            onset = n
            duration = 1
            p_prev = p

    # add last note
    if p_prev > 0:
        # add note
        duration_sec = duration * hop / float(fs)
        onset_sec = onset * hop / float(fs)
        notes.append((onset_sec, duration_sec, p_prev))

    return notes

    
def hz2midi(hz):
    '''
    Convert hz to midi note
    '''
    
    # convert from Hz to midi note
    hz_nonneg = hz.copy()
    idx = hz_nonneg <= 0
    hz_nonneg[idx] = 1
    midi = 69 + 12*np.log2(hz_nonneg/440.)
    midi[idx] = 0

    # round
    midi = np.round(midi)

    return midi
    

def melosynth(times, freqs, bpm, orig_url, outputfolder, outputid, 
              nHarmonics=1, square=False, useneg=False,
              fs=44100, hop=128, smooth=0.25, minduration=0.1):
    '''
    Load pitch sequence from  a txt/csv file and synthesize it into a .wav

    :parameters:
    - times: np.ndarray
    Array of timestamps (float)

    - freqs: np.ndarray
    Array of corresponding frequency values (float)
    
    - orig_url: str
    URL string of the original song. Used to write to 
    [outputid].melosynth.mix.wav

    - outputfolder: str
    Path to where output csv and wav files should be saved.

    - outputid: str
    [outputid].csv and [outputid].melosynth.wav will be written.

    - fs : int
    Sampling frequency for the synthesized file.

    - nHarmonics : int
    Number of harmonics (including the fundamental) to use in the synthesis
    (default is 1). As the number is increased the wave will become more
    sawtooth-like.

    - square : bool
    When set to true, the waveform will converge to a square wave instead of
    a sawtooth as the number of harmonics is increased.

    - useneg : bool
    By default, negative frequency values (unvoiced frames) are synthesized as
    silence. If useneg is set to True, these frames will be synthesized using
    their absolute values (i.e. as voiced frames).
    '''

    # Preprocess input parameters
    fs = int(float(fs))
    nHarmonics = int(nHarmonics)

    # Load pitch sequence
    logging.info('Loading data...')
    #times, freqs = loadmel(inputfile)

    # Preprocess pitch sequence
    if useneg:
        freqs = np.abs(freqs)
    else:
        freqs[freqs < 0] = 0
    # Impute silence if start time > 0
    if times[0] > 0:
        estimated_hop = np.median(np.diff(times))
        prev_time = max(times[0] - estimated_hop, 0)
        times = np.insert(times, 0, prev_time)
        freqs = np.insert(freqs, 0, 0)


    logging.info('Generating wave...')
    signal = []

    translen = 0.010 # duration (in seconds) for fade in/out and freq interp
    phase = np.zeros(nHarmonics) # start phase for all harmonics
    f_prev = 0 # previous frequency
    t_prev = 0 # previous timestamp
    for t, f in zip(times, freqs):

        # Compute number of samples to synthesize
        nsamples = int(np.round((t - t_prev) * fs))

        if nsamples > 0:
            # calculate transition length (in samples)
            translen_sm = float(min(np.round(translen*fs), nsamples))

            # Generate frequency series
            nsamples
            freq_series = np.ones(nsamples) * f_prev

            # Interpolate between non-zero frequencies
            if f_prev > 0 and f > 0:
                freq_series += np.minimum(np.arange(nsamples)/translen_sm, 1) *\
                               (f - f_prev)
            elif f > 0:
                freq_series = np.ones(nsamples) * f

            # Repeat for each harmonic
            samples = np.zeros(nsamples)
            for h in range(nHarmonics):
                # Determine harmonic num (h+1 for sawtooth, 2h+1 for square)
                hnum = 2*h+1 if square else h+1
                # Compute the phase of each sample
                phasors = 2 * np.pi * (hnum) * freq_series / float(fs)
                phases = phase[h] + np.cumsum(phasors)
                # Compute sample values and add
                samples += np.sin(phases) / (hnum)
                # Update phase
                phase[h] = phases[-1]

            # Fade in/out and silence
            if f_prev == 0 and f > 0:
                samples *= np.minimum(np.arange(nsamples)/translen_sm, 1)
            if f_prev > 0 and f == 0:
                samples *= np.maximum(1 - (np.arange(nsamples)/translen_sm), 0)
            if f_prev == 0 and f == 0:
                samples *= 0

            # Append samples
            signal.extend(samples)

        t_prev = t
        f_prev = f

    # Normalize signal
    signal = np.asarray(signal)
    signal *= 0.8 / float(np.max(signal))
    
    '''
    Convert Synthesized Melody to Midi
        
    '''

    # impute missing 0's to compensate for starting timestamp
    #freqs = np.insert(freqs, 0, [0]*8)
    
    # convert f0 to midi notes
    logging.info("Converting Hz to MIDI notes...")
    midi_signal = hz2midi(freqs)

    # segment sequence into individual midi notes
    notes = midi_to_notes(midi_signal, fs, hop, smooth, minduration)
    
    '''
    Save to output files
        Save timestamp and frequency data to .csv.
        Save synthesized melody to .wav.
        Save synthesized melody and original audio mix to .wav.
        Save synthesized melody to .midi.
    '''

    logging.info('Saving csv file...')
    # music/data/melodia/data
    csv_data = np.array([times, freqs]).T
    csv_f = os.path.join(outputfolder, "data", outputid + ".csv")
    np.savetxt(csv_f, csv_data, delimiter=",")
    
    logging.info('Saving wav file...')
    # music/data/melodia/synths
    wav_f = os.path.join(outputfolder, "synths", outputid + ".melo.wav")
    wavwrite(np.asarray(signal), wav_f, fs)
    
    logging.info('Saving mixed wav file...')
    # music/data/melodia/synths
    wav_mix_f = os.path.join(outputfolder, "synths", outputid + ".melo.and.orig.wav")
    
    # cut off last second to keep both arrays same size
    orig, sr = librosa.load(orig_url, sr=44100, mono=True, duration=29, offset=0.05)
    melo, sr = librosa.load(wav_f, sr=44100, mono=True, duration=29)
    orig_and_melo = np.add(orig*0.3, melo*0.7)
    wavwrite(orig_and_melo, wav_mix_f, fs)
    
    logging.info('Saving original wav file...')
    # music/data/melodia/synths
    wav_orig_f = os.path.join(outputfolder, "synths", outputid + ".orig.wav")
    wavwrite(orig, wav_orig_f, fs)
    
    logging.info('Saving midi file...')
    # music/data/melodia/synths
    midi_f = os.path.join(outputfolder, "synths", outputid + ".melo.midi")
    midiwrite(midi_f, notes, bpm)
    
    # Return csv_f, wav_f, wav_mix_f, wav_orig_f, midi_f
    return(csv_f, wav_f, wav_mix_f, wav_orig_f, midi_f)