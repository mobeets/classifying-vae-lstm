import cPickle
import argparse
import numpy as np
from music21 import key, stream, note, chord, interval, scale
from progress.bar import Bar

def relative_major(k):
    return k if k.isupper() else key.Key(k).relative.tonicPitchNameWithCase

def intervalToKey(k, keyGoal):
    """
    note: 'C' is Cmaj, 'c' is Cmin
    """
    k_goal_main = key.Key(keyGoal)
    k_goal_rel = k_goal_main.relative
    k_goal = k_goal_main if k.mode == k_goal_main.mode else k_goal_rel
    return interval.Interval(k.tonic, k_goal.tonic)

def key_strength(counts, keyGoal):
    sc = scale.MajorScale(keyGoal) if keyGoal.isupper() else scale.MinorScale(keyGoal)
    gs = [n.pitchClass for n in sc.getPitches()][:-1]
    return 1.0*counts[gs].sum()/counts.sum()

def song_to_music21(notes):
    song = stream.Stream()
    for nums in notes:
        song.append(chord.Chord(nums))
    return song

def analyze_key(notes):
    song = song_to_music21(notes)
    return song.analyze('key')

def findOffsetToStandardize(curKey, keyGoal):
    i = intervalToKey(curKey, keyGoal)
    return i.chromatic.semitones, curKey.mode == key.Key(keyGoal).mode

def prepRandomKeys(keyGoal, n):
    minKeyOpts = [key for key in keyGoal if key.islower()]
    majKeyOpts = [key for key in keyGoal if key.isupper()]
    if not minKeyOpts:
        majKeyGoals = np.random.choice(majKeyOpts, (n,))
        minKeyGoals = majKeyGoals
    elif not majKeyOpts:
        minKeyGoals = np.random.choice(minKeyOpts, (n,))
        majKeyGoals = minKeyGoals
    else:
        minKeyGoals = np.random.choice(minKeyOpts, (n,))
        majKeyGoals = np.random.choice(majKeyOpts, (n,))
    return minKeyGoals, majKeyGoals

def convert_song_keys(train_file, outfile=None,
    keyGoal='C', enforceMode=False, doRandomize=False, verbose=True, minVal=21, maxVal=108):
    """
    keyGoal is key to convert songs to ('C' is Cmaj, 'c' is Cmin)
        - this can contain multiple options
        - e.g., 'CcDd' will convert songs in a major key to Cmaj or Dmaj, and songs in a minor key to Cmin or Dmin
    if enforceMode is True, will ignore songs that don't match the mode
        i.e., major vs. minor
        otherwise, will convert to the relative key (e.g., Cmaj -> Amin)
    """
    D = cPickle.load(open(train_file))
    E = {}
    # split key goals on spaces or commas
    keyOpts = keyGoal.replace(' ', ',').split(',')
    for k in D:
        E[k] = []
        E[k + '_mode'] = []
        E[k + '_key'] = []
        minKeyGoals, majKeyGoals = prepRandomKeys(keyGoal, len(D[k]))
        bar = Bar('Processing', max=len(D[k]))
        for i, song in enumerate(D[k]):
            bar.next()
            curKey = analyze_key(song)
            if curKey.mode == 'minor':
                curKeyGoal = minKeyGoals[i]
            elif curKey.mode == 'major':
                curKeyGoal = majKeyGoals[i]
            else:
                assert False
            offset, modeMatch = findOffsetToStandardize(curKey, curKeyGoal)
            if enforceMode and not modeMatch:
                if verbose:
                    print "{} -> [removed]".format(curKey.name)
                continue
            song_new = [[n+offset for n in notes] for notes in song]

            # make sure notes are in range
            all_notes = [y for x in song for y in x]
            if min(all_notes) < minVal:
                offset += 12
                song_new = [[n+offset for n in notes] for notes in song]
            elif max(all_notes) > maxVal:
                offset -= 12
                song_new = [[n+offset for n in notes] for notes in song]

            if verbose:
                print "{} -> {}".format(curKey.name, curKeyGoal)
            E[k].append(song_new)
            E[k + '_mode'].append(curKeyGoal.isupper())
            E[k + '_key'].append(curKeyGoal)
        bar.finish()
    if outfile is not None:
        write_pickle(E, outfile)
    return E

def write_pickle(D, outfile):
    with open(outfile, 'wb') as f:
        cPickle.dump(D, f, protocol=cPickle.HIGHEST_PROTOCOL)

def print_keys_summary(keys, isMajs):
    keys = np.array(keys)
    isMajs = np.array(isMajs)
    for ky in np.unique(keys):
        c1 = ((keys == ky) & isMajs).sum()
        c2 = ((keys == ky) & ~isMajs).sum()
        if c1 > 0:
            print ky + 'maj', c1
        if c2 > 0:
            print ky + 'min', c2

def print_songs_key_summary(train_file, verbose=True):
    D = cPickle.load(open(train_file))
    for k in D:
        keys = []
        isMajs = []
        if k.endswith('_mode') or k.endswith('_key'):
            continue
        for song in D[k]:
            key = analyze_key(song)
            if verbose:
                print key
            keys.append(key.tonic.name)
            isMajs.append(key.mode == 'major')
        if verbose:
            print
        print '==========='
        print k
        print '==========='
        print_keys_summary(keys, isMajs)
        if verbose:
            print

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,
                help='input file (.pickle)')
    parser.add_argument('-o', '--output_file', type=str, default='',
                help='output file (.pickle)')
    parser.add_argument('--key_goal', type=str, default='C',
                help='key(s) to transpose songs (uppercase for major, lowercase for minor); if song has a different mode, will transpose to the relative minor/major; e.g. "C D E"')
    parser.add_argument("--same_mode", action="store_true", 
                help="only keep songs that have the same mode (i.e., major vs. minor)")
    parser.add_argument("--randomize", action="store_true", 
                help="choose random keys for each song")
    parser.add_argument('-v', "--verbose", action="store_true", 
                help="display everything")
    parser.add_argument("--summary_only", action="store_true", 
                help="print a summary of the keys present in the input_file (no outputs)")
    args = parser.parse_args()

    if args.summary_only:
        print_songs_key_summary(args.input_file, verbose=args.verbose)
    else:
        # if args.randomize and len(args.key_goal) == 1:
        #     print "Cannot randomize from only one option!"
        if args.output_file == '':
            assert False, "You must specify an output file!"
        convert_song_keys(args.input_file,
            outfile=args.output_file,
            keyGoal=args.key_goal,
            enforceMode=args.same_mode,
            doRandomize=args.randomize,
            verbose=args.verbose)
