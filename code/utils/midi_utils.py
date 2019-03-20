"""
source: https://github.com/yoavz/music_rnn
"""
from __future__ import division

import sys, os
from collections import defaultdict
import numpy as np
import midi

import sys
import argparse
import numpy as np
import pretty_midi
import librosa

RANGE = 128

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

class MidiWriter(object):

    def __init__(self, verbose=False, default_vel=100):
        self.verbose = verbose
        self.note_range = RANGE
        self.default_velocity = default_vel

    def note_off(self, val, tick):
        self.track.append(midi.NoteOffEvent(tick=tick, pitch=val))
        return 0

    def note_on(self, val, tick):
        self.track.append(midi.NoteOnEvent(tick=tick, pitch=val, 
            velocity=self.default_velocity))
        return 0

    def dump_sequence_to_midi(self, seq, output_filename,
        time_step=120, resolution=480, metronome=24, offset=21,
        format='final'):
        
        n_times, n_keys = seq.shape
        piano_roll = np.zeros((128, n_times))
        piano_roll[offset:offset+n_keys] = seq.T

        # time_step == 120 bpm?
        # resolution == n_steps to take?
        # metronome??
        # format??
        pm = piano_roll_to_pretty_midi(piano_roll, fs=resolution//time_step)

        pm.write(output_filename)
    '''
    def dump_sequence_to_midi_orig(self, seq, output_filename,
        time_step=120, resolution=480, metronome=24, offset=21,
        format='final'):
        
        if self.verbose:
            print("Dumping sequence to MIDI file: {}".format(output_filename))
            print("Resolution: {}".format(resolution))
            print("Time Step: {}".format(time_step))

        pattern = midi.Pattern(resolution=resolution)
        self.track = midi.Track()
        # metadata track
        meta_track = midi.Track()
        
        time_sig = midi.TimeSignatureEvent()
        time_sig.set_numerator(4)
        time_sig.set_denominator(4)
        time_sig.set_metronome(metronome)
        time_sig.set_thirtyseconds(8)

        meta_track.append(time_sig)
        pattern.append(meta_track)
        
        # reshape to (SEQ_LENGTH X NUM_DIMS)
        if format == 'icml':
            # assumes seq is list of lists, where each inner list are all the midi notes that were non-zero at that given timestep
            sequence = np.zeros([len(seq), self.note_range])
            sequence = [1 if i in tmstp else 0 for i in range(self.note_range) for tmstp in seq]
            sequence = np.reshape(sequence, [self.note_range,-1]).T
        elif format == 'flat':
            sequence = np.reshape(seq, [-1, self.note_range])
        else:
            sequence = seq

        time_steps = sequence.shape[0]
        if self.verbose:
            print("Total number of time steps: {}".format(time_steps))
        
        tick = time_step
        self.notes_on = { n: False for n in range(self.note_range) }
        # for seq_idx in range(188, 220):
        for seq_idx in range(time_steps):
            notes = np.nonzero(sequence[seq_idx, :])[0].tolist()
            print(notes)
            # n.b. notes += 21 ??
            # need to be in range 21,109
            notes = [n+offset for n in notes]
            print(notes)

            # this tick will only be assigned to first NoteOn/NoteOff in
            # this time_step

            # NoteOffEvents come first so they'll have the tick value
            # go through all notes that are currently on and see if any
            # turned off
            for n in self.notes_on:
                if self.notes_on[n] and n not in notes:
                    tick = self.note_off(n, tick)
                    self.notes_on[n] = False

            # Turn on any notes that weren't previously on
            for note in notes:
                if not self.notes_on[note]:
                    tick = self.note_on(note, tick)
                    self.notes_on[note] = True

            

            tick += time_step

        # flush out notes
        for n in self.notes_on:
            if self.notes_on[n]:
                self.note_off(n, tick)
                tick = 0
                self.notes_on[n] = False

        pattern.append(self.track)
        midi.write_midifile(output_filename, pattern)
    '''
def write_sample(sample, outdir, fnm, isHalfAsSlow=False):
    if isHalfAsSlow:
        sample = np.repeat(sample, 2, axis=0)
    fnm = os.path.join(outdir, fnm + '.mid')
    MidiWriter().dump_sequence_to_midi(sample, fnm)
