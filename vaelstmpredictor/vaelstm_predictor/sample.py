import os.path
import argparse
import numpy as np
from keras.utils import to_categorical

from ..utils.data_utils import PianoData
from ..utils.midi_utils import write_sample
from .model import load_model, generate_sample, make_decoder
from .model import make_w_encoder, make_z_encoder

def gen_samples(P, dec_model, w_enc_model, z_enc_model, args, margs):
    key_map = {v: k for k, v in P.key_map.iteritems()}
    inds = np.arange(len(P.test_song_keys))
    if args.c is not None: # user set key
        kys = np.array([key_map[k] for k in P.test_song_keys])
        ix = (kys == args.c)
        inds = inds[ix]
    np.random.shuffle(inds)
    outfile = lambda j,i: args.run_name + '_' + str(j)    
    outfile_seed = lambda j,i: args.run_name + str(j) + '_seed_' + str(i)
    for j, i in enumerate(inds[:args.n]):
        cur_key_ind = P.test_song_keys[i]
        w_val = None if args.infer_w else to_categorical(cur_key_ind, margs['n_classes'])
        x_seed = P.x_test[i]
        sample = generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, args.t, margs['use_x_prev'], w_val=w_val, w_discrete=args.discrete_w, seq_length=margs['seq_length'])
        
        write_sample(sample, args.sample_dir, outfile(j,i),
            'jsb' in args.train_file.lower())
        write_sample(x_seed, args.sample_dir, outfile_seed(j,i),
            'jsb' in args.train_file.lower())

def sample(args):
    # load models
    train_model, _, margs = load_model(args.model_file, optimizer='adam')
    w_enc_model = make_w_encoder(train_model, margs['original_dim'],
        margs['n_classes'], margs['seq_length'])
    z_enc_model = make_z_encoder(train_model, margs['original_dim'],
        margs['n_classes'], (margs['intermediate_dim'], margs['latent_dim']))
    dec_model = make_decoder(train_model, margs['original_dim'],
        margs['intermediate_dim'], margs['latent_dim'], margs['n_classes'],
        margs['use_x_prev'])

    # load data
    P = PianoData(args.train_file,
        batch_size=1,
        seq_length=args.t,
        squeeze_x=False)

    gen_samples(P, dec_model, w_enc_model, z_enc_model, args, margs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                        help='tag for current run')
    parser.add_argument("--infer_w", action="store_true", 
                        help="infer w when generating")
    parser.add_argument("--discrete_w", action="store_true", 
                        help="sample discrete w when generating")
    parser.add_argument('-t', type=int, default=32,
                        help='number of timesteps per sample')
    parser.add_argument('-n', type=int, default=1,
                        help='number of samples')
    parser.add_argument('-c', type=str,
                        help='set key of seed sample')
    parser.add_argument('--sample_dir', type=str,
                        default='../data/samples',
                        help='basedir for saving output midi files')
    parser.add_argument('-i', '--model_file', type=str, default='',
                        help='preload model weights (no training)')
    parser.add_argument('--train_file', type=str,
                        default='../data/input/JSB Chorales_Cs.pickle',
                        help='file of training data (.pickle)')
    args = parser.parse_args()
    sample(args)
    # $ brew install timidity
    # $ timidity filename.mid
