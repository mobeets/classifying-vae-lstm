import argparse
import numpy as np
from keras.utils import to_categorical
from .utils.pianoroll import PianoData
from .utils.midi_utils import write_sample
from model import load_model, generate_sample, make_decoder, make_w_encoder, make_z_encoder, sample_z

def make_sample(P, dec_model, w_enc_model, z_enc_model, args, margs):
    # generate and write sample
    seed_ind = np.random.choice(xrange(len(P.x_test)))
    x_seed = P.x_test[seed_ind][0]
    seed_key_ind = P.test_song_keys[seed_ind]
    w_val = None if args.infer_w else to_categorical(seed_key_ind, margs['n_classes'])
    sample = generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, args.t,  w_val=w_val, use_z_prior=args.use_z_prior, use_x_prev=margs['use_x_prev'])
    write_sample(sample, args.sample_dir, args.run_name, True)

def sample(args):
    # load models
    train_model, enc_model, margs = load_model(args.model_file, no_x_prev=args.no_x_prev)
    w_enc_model = make_w_encoder(train_model, margs['original_dim'])
    z_enc_model = make_z_encoder(train_model, margs['original_dim'], margs['n_classes'], (margs['intermediate_dim'], margs['latent_dim']))
    dec_model = make_decoder(train_model, (margs['intermediate_dim'], margs['latent_dim']), margs['n_classes'], use_x_prev=margs['use_x_prev'])

    # load data
    P = PianoData(args.train_file,
        batch_size=1,
        seq_length=args.t,
        squeeze_x=True)

    basenm = args.run_name
    for i in xrange(args.n):
        args.run_name = basenm + '_' + str(i)
        make_sample(P, dec_model, w_enc_model, z_enc_model, args, margs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                        help='tag for current run')
    parser.add_argument("-n", type=int, default=1,
                        help="number of samples")
    parser.add_argument("--use_z_prior", action="store_true", 
                        help="sample z from standard normal at each timestep")
    parser.add_argument('-t', type=int, default=32,
                        help='number of timesteps per sample')
    parser.add_argument("--infer_w", action="store_true", 
                        help="infer w when generating")
    parser.add_argument("--no_x_prev", action="store_true", 
                        help="override use_x_prev")
    parser.add_argument('--sample_dir', type=str,
                        default='../data/samples',
                        help='basedir for saving output midi files')
    parser.add_argument('--model_dir', type=str,
                        default='../data/models',
                        help='basedir for saving model weights')
    parser.add_argument('-i', '--model_file', type=str, default='',
                        help='preload model weights (no training)')
    parser.add_argument('--train_file', type=str,
                        default='../data/input/JSB Chorales_Cs.pickle',
                        help='file of training data (.pickle)')
    args = parser.parse_args()
    sample(args)
    # $ brew install timidity
    # $ timidity filename.mid
