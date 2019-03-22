import argparse
import numpy as np

from collections import namedtuple
from json import load as json_load

from ..utils.pianoroll import PianoData
from ..utils.midi_utils import write_sample

# from .model import load_model, generate_sample, make_decoder
# from .model import make_w_encoder, make_z_encoder, sample_z
from .model import VAEClassifier
"""
Code to load pianoroll data (.pickle)
"""
import numpy as np
'''
def make_sample(P, dec_model, w_enc_model, z_enc_model, args, margs):
    # generate and write sample
    seed_ind = np.random.choice(list(range(len(P.x_test))))
    x_seed = P.x_test[seed_ind][0]
    seed_key_ind = P.test_song_keys[seed_ind]
    w_val = None if args.infer_w else to_categorical(seed_key_ind, margs['n_classes'])
    sample = generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, 
                                args.t,  w_val=w_val, 
                                use_z_prior=args.use_z_prior, 
                                use_x_prev=margs['use_x_prev'])

    print('[INFO] Storing New MIDI file in {}/{}.mid'.format(
                                args.sample_dir, args.run_name))
    write_sample(sample, args.sample_dir, args.run_name, True)
'''
def sample(args):
    json_input = json_load(open(args.model_file.replace('.h5', '.json')))
    
    json_converter = namedtuple('json_converter', json_input.keys())
    margs = json_converter(**json_input)

    vae_dims = (margs.intermediate_dim, margs.latent_dim)
    classifier_dims = (margs.intermediate_class_dim, margs.n_classes)
    
    vae_clf = VAEClassifier(batch_size = margs.batch_size,
                            original_dim = margs.original_dim, 
                            vae_dims = vae_dims,
                            classifier_dims = classifier_dims, 
                            optimizer = margs.optimizer.replace('-wn',''),
                            clf_weight = margs.clf_weight, 
                            use_prev_input = margs.use_prev_input)
    
    vae_clf.get_model()
    vae_clf.model.load_weights(args.model_file)
    
    # load data
    P = PianoData(args.train_file, seq_length=args.t, squeeze_x=True)
    
    basenm = args.run_name
    for i in range(args.n):
        args.run_name = basenm + '_' + str(i)
        vae_clf.make_sample(P, args)
