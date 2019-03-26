import argparse

from vaelstmclassifier.utils.data_utils import PianoData
from vaelstmclassifier.vae_classifier import sample
vae_classifier_sample = sample.sample # rename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                        help='tag for current run')
    parser.add_argument("-n", "--num_samples", type=int, default=1,
                        help="number of samples")
    parser.add_argument("--use_latent_prior", action="store_true", 
                        help="sample z from standard normal at each timestep")
    parser.add_argument('-t', type=int, default=32,
                        help='number of timesteps per sample')
    parser.add_argument("--infer_w", action="store_true", 
                        help="infer w when generating")
    parser.add_argument("--no_x_prev", action="store_true", 
                        help="override use_x_prev")
    parser.add_argument('--sample_dir', type=str,
                        default='data/samples',
                        help='basedir for saving output midi files')
    parser.add_argument('--model_dir', type=str,
                        default='data/models',
                        help='basedir for saving model weights')
    parser.add_argument('-i', '--model_file', type=str, default='',
                        help='preload model weights (no training)')
    parser.add_argument('--train_file', type=str,
                        default='data/input/JSB Chorales_Cs.pickle',
                        help='file of training data (.pickle)')
    
    args = parser.parse_args()
    
    P = PianoData(args.train_file, seq_length=args.t, squeeze_x=True)

    vae_classifier_sample(args, data_instance = P)

    # $ brew install timidity
    # $ timidity filename.mid
