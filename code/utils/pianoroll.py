"""
Code to load pianoroll data (.pickle)
data source: http://www-etud.iro.umontreal.ca/~boulanni/icml2012
"""
import numpy as np
import cPickle
from preprocess import relative_major, analyze_key, key_strength

def assess_key_strength(sample, keyGoal):
    counts = np.roll(np.reshape(np.hstack([sample.sum(axis=0), np.zeros((8,))]), (8,12)).sum(axis=0), -3)
    return key_strength(counts, keyGoal)

def pianoroll_to_song(roll, offset=21):
    f = lambda x: (np.where(x)[0]+offset).tolist()
    return [f(s) for s in roll]

def song_to_pianoroll(song, offset=21):
    """
    song = [(60, 72, 79, 88), (72, 79, 88), (67, 70, 76, 84), ...]
    """
    rolls = []
    all_notes = [y for x in song for y in x]
    if min(all_notes)-offset < 0:
        offset -= 12
        # assert False
    if max(all_notes)-offset > 87:
        offset += 12
        # assert False
    for notes in song:
        roll = np.zeros(88)
        roll[[n-offset for n in notes]] = 1.
        rolls.append(roll)
    return np.vstack(rolls)

def sliding_inds(n, seq_length, step_length):
    return np.arange(n-seq_length, step=step_length)

def sliding_window(roll, seq_length, step_length=1):
    """
    returns [n x seq_length x 88]
        if step_length == 1, then roll[i,1:] == roll[i+1,:-1]
    """
    rolls = []
    for i in sliding_inds(roll.shape[0], seq_length, step_length):
        rolls.append(roll[i:i+seq_length,:])
    if len(rolls) == 0:
        return np.array([])
    return np.dstack(rolls).swapaxes(0,2).swapaxes(1,2)

def songs_to_pianoroll(songs, seq_length, step_length, inner_fcn=song_to_pianoroll):
    """
    songs = [song1, song2, ...]
    """
    rolls = [sliding_window(inner_fcn(s), seq_length, step_length) for s in songs]
    rolls = [r for r in rolls if len(r) > 0]
    inds = [i*np.ones((len(r),)) for i,r in enumerate(rolls)]
    return np.vstack(rolls), np.hstack(inds)

class PianoData:
    def __init__(self, train_file, batch_size=None, seq_length=1, step_length=1, return_y_next=True, return_y_hist=False, squeeze_x=True, squeeze_y=True, use_rel_major=True):
        """
        returns [n x seq_length x 88] where rows referring to the same song will overlap an amount determined by step_length

        specifying batch_size will ensure that that mod(n, batch_size) == 0
        """
        D = cPickle.load(open(train_file))        
        self.train_file = train_file # .pickle source file
        self.batch_size = batch_size # ensures that nsamples is divisible by this
        self.seq_length = seq_length # returns [n x seq_length x 88]
        self.step_length = step_length # controls overlap in rows of X
        self.return_y_next = return_y_next # if True, y is next val of X; else y == X
        self.return_y_hist = return_y_hist # if True, y is next val of X for each column of X; else y == [n x 1 x 88]
        self.squeeze_x = squeeze_x # remove singleton dimensions in X?
        self.squeeze_y = squeeze_y # remove singleton dimensions in y?
        self.use_rel_major = use_rel_major # minor keys get mapped to their relative major, e.g. 'a' -> 'C'

        # sequences with song indices
        self.x_train, self.y_train, self.train_song_inds = self.make_xy(D['train'])
        self.x_test, self.y_test, self.test_song_inds = self.make_xy(D['test'])
        self.x_valid, self.y_valid, self.valid_song_inds = self.make_xy(D['valid'])

        # # song index per sequence
        # self.train_song_inds = self.song_inds(D['train'])
        # self.test_song_inds = self.song_inds(D['test'])
        # self.valid_song_inds = self.song_inds(D['valid'])

        # mode per sequence
        if 'train_mode' in D:
            self.train_song_modes = self.song_modes(D['train_mode'], self.train_song_inds)
            self.test_song_modes = self.song_modes(D['test_mode'], self.test_song_inds)
            self.valid_song_modes = self.song_modes(D['valid_mode'], self.valid_song_inds)
        if 'train_key' in D:
            D = self.update_keys(D)
            self.key_map = self.make_keymap(D)
            self.train_song_keys = self.song_keys(D['train_key'], self.train_song_inds)
            self.test_song_keys = self.song_keys(D['test_key'], self.test_song_inds)
            self.valid_song_keys = self.song_keys(D['valid_key'], self.valid_song_inds)

    def make_xy(self, songs):
        inner_fcn = song_to_pianoroll
        x_rolls, song_inds = songs_to_pianoroll(songs, self.seq_length + int(self.return_y_next), self.step_length, inner_fcn=inner_fcn)
        x_rolls = self.adjust_for_batch_size(x_rolls)
        song_inds = self.adjust_for_batch_size(song_inds)
        if self.return_y_next: # make Y the last col of X
            if self.return_y_hist:
                y_rolls = x_rolls[:,1:,:]
            else:
                y_rolls = x_rolls[:,-1,:]
            x_rolls = x_rolls[:,:-1,:]
        else:
            y_rolls = x_rolls
        if self.squeeze_x: # e.g., if X is [n x 1 x 88]
            x_rolls = x_rolls.squeeze()
        if self.squeeze_y:
            y_rolls = y_rolls.squeeze()
        return x_rolls, y_rolls, song_inds

    def song_modes(self, modes, song_inds):
        return np.array(modes)[song_inds.astype(int)]

    def update_keys(self, D):
        if not self.use_rel_major:
            return
        D['train_key'] = [relative_major(k) for k in D['train_key']]
        D['test_key'] = [relative_major(k) for k in D['test_key']]
        D['valid_key'] = [relative_major(k) for k in D['valid_key']]
        return D

    def make_keymap(self, D):
        all_keys = np.unique(np.hstack([D['train_key'], D['test_key'], D['valid_key']]))
        return dict(zip(all_keys, xrange(len(all_keys))))

    def song_keys(self, keys, song_inds):
        """
        also converts keys to ints, e.g., ['A', 'B', 'C'] -> [0, 1, 2]
        """
        key_inds = [self.key_map[k] for k in keys]
        return np.array(key_inds)[song_inds.astype(int)]

    def adjust_for_batch_size(self, items):
        if self.batch_size is None:
            return items
        mod = (items.shape[0] % self.batch_size)
        return items[:-mod] if mod > 0 else items

def reanalyze_key(P):
    """
    if we were to analyze the key of each data point, how consistent would it be with the key of the entire song? this suggests an upper-bound for how well we will be able to classify key with this data
    """
    from progress.bar import Bar
    invkey = {v:k for k,v in P.key_map.iteritems()}
    D = {}
    for ky in ['x_train', 'x_valid', 'x_test']:
        keys = []
        scs = []
        new_keys = []
        rolls = getattr(P, ky)
        inds = np.arange(len(rolls))
        bar = Bar('Processing', max=len(inds))
        # np.random.shuffle(inds)
        for i in inds:
            song = pianoroll_to_song(rolls[i])
            k = analyze_key(song).tonicPitchNameWithCase
            new_keys.append(k)
            bar.next()
            continue
            k0 = getattr(P, ky[2:] + '_song_keys')[i]
            keys.append((k0, P.key_map.get(k, 'NONE')))
            sc = key_strength(song, invkey[k0])
            scs.append(sc)
            if i % 50 == 0:
                # mean accuracy, score with original key
                print (np.array(keys)[:,0] == np.array(keys)[:,1]).mean(), np.array(scs).mean()
        D[ky] = rolls
        D[ky + '_keys'] = np.array(new_keys)
        bar.finish()
    return D

def load_npz_to_update_keys(infile, P, use_rel_major=True):
    D = np.load(infile)
    all_keys = []
    E = {}
    for ky in D:
        maxind = getattr(P, ky.replace('_keys', '')).shape[0]
        if not ky.endswith('_keys'):
            assert((D[ky][:maxind] == getattr(P, ky)).all())
            continue
        E[ky] = D[ky][:maxind]
        if use_rel_major:
            E[ky] = [relative_major(k) for k in E[ky]]
        all_keys.append(E[ky])
    all_keys = np.unique(np.hstack(all_keys))
    P.key_map = dict(zip(all_keys, xrange(len(all_keys))))
    for ky in E:
        keys = np.array([P.key_map[k] for k in E[ky]])
        setattr(P, ky.split('_')[1] + '_song_keys', keys)
    return P    

if __name__ == '__main__':
    train_file = '../data/input/Piano-midi_all.pickle'
    P = PianoData(train_file, seq_length=1, step_length=1)
