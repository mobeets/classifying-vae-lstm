import argparse
import os

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmpredictor.vae_predictor.train import train_vae_predictor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, # default='run_',
                help='tag for current run')
    parser.add_argument('--predictor_type', type=str, default="classification",
                help='select `classification` or `regression`')
    parser.add_argument('--batch_size', type=int, default=128,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=0,
                help='input dim')
    parser.add_argument('--vae_hidden_dim', type=int, default=128,
                help='intermediate dim')
    parser.add_argument('--vae_latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--seq_length', type=int, default=1,
                help='sequence length (concat)')
    parser.add_argument('--predictor_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--prediction_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument('--predictor_hidden_dim', type=int, default=128,
                help='intermediate dims for class/regr predictor')
    parser.add_argument('--predictor_latent_dim', type=int, default=0,
                help='predictor dims for class/regr prediction')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--do_chckpt", action="store_true",
                help="save model checkpoints")
    parser.add_argument("--predict_next", action="store_true", 
                help="use state_now to 'autoencode' state_next")
    parser.add_argument("--use_prev_input", action="store_true",
                help="use state_prev to help latent_now decode state_now")
    parser.add_argument('--patience', type=int, default=10,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str, default='data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str,
                default='data/input/JSB Chorales_Cs.pickle',
                help='file of training data (.pickle)')
    parser.add_argument('--no_squeeze_x', action="store_true",
                help='whether to squeeze the x dimension')
    parser.add_argument('--no_squeeze_y', action="store_true",
                help='whether to squeeze the x dimension')
    parser.add_argument('--step_length', type=int, default=1,
                help="Length of the step for overlap in song(s)")
    parser.add_argument('--data_type', type=str, default='piano',
                help="The type of data to fit ['piano', 'mnist', 'exoplanet']")
    parser.add_argument('--debug', action="store_true",
                help="if debug; then stop before model.fit")
    args = parser.parse_args()
    
    if 'class' in args.predictor_type.lower():
        args.predictor_type = 'classification'
    if 'regr' in args.predictor_type.lower():
        args.predictor_type = 'regression'

    if args.predictor_type is 'regression': args.n_labels = 1

    data_types = ['piano', 'mnist', 'exoplanet']
    
    if 'piano' in args.data_type.lower():
        from vaelstmpredictor.utils.data_utils import PianoData

        args.data_type = 'PianoData'

        return_label_next = args.predict_next or args.use_prev_input

        P = PianoData(train_file = args.train_file,
                      batch_size = args.batch_size,
                      seq_length = args.seq_length,
                      step_length=args.step_length,
                      return_label_next = return_label_next,
                      squeeze_x = not args.no_squeeze_x,
                      squeeze_y = not args.no_squeeze_y)

        # Keep default unless modified inside `PianoData` instance
        args.original_dim = P.original_dim or args.original_dim

        data_instance = P

    elif 'mnist' in args.data_type.lower():
        from vaelstmpredictor.utils.data_utils import MNISTData

        args.data_type = 'MNIST'
        data_instance = MNISTData(batch_size = args.batch_size)

    elif 'exoplanet' in args.data_type.lower():
        from vaelstmpredictor.utils.data_utils import ExoplanetData

        args.data_type = 'ExoplanetSpectra'
        data_instance = ExoplanetData(train_file = args.train_file,
                                      batch_size = args.batch_size)
    else:
        raise ValueError("`data_type` must be in list {}".format(data_types))

    n_train, n_features = data_instance.data_train.shape
    n_test, n_features = data_instance.data_valid.shape

    if args.original_dim is 0: args.original_dim = n_features
    
    time_stmp = int(time())
    args.run_name = '{}_{}_{}'.format(args.run_name, args.data_type, time_stmp)

    print('\n\n[INFO] Run Base Name: {}\n'.format(args.run_name))

    vae_model, best_loss, history = train_vae_predictor(clargs = args, 
                                                data_instance = data_instance)

    print('\n\n[INFO] The Best Loss: {}\n'.format(best_loss))
    joblib_save_loc = '{}/{}_trained_model_output.joblib.save'.format(
                                                            args.model_dir,
                                                            args.run_name)

    weights_save_loc = '{}/{}_trained_model_weights.save'.format(
                                                            args.model_dir,
                                                            args.run_name)
    
    model_save_loc = '{}/{}_trained_model_full.save'.format(args.model_dir,
                                                            args.run_name)
    
    vae_model.model.save_weights(weights_save_loc, overwrite=True)
    vae_model.model.save(model_save_loc, overwrite=True)
    
    joblib.dump({'best_loss':best_loss,'history':history}, joblib_save_loc)