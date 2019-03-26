import argparse
import os

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmclassifier.utils.pianoroll import PianoData
from vaelstmclassifier.vae_classifier.train import train_vae_classifier
# vae_classifier_train = train.train_vae_classifier # rename

class BlankClass(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, # default='run_',
                help='tag for current run')
    parser.add_argument('--network_type', type=str, default="classification",
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
    parser.add_argument('--prediction_hidden_dim', type=int, default=128,
                help='intermediate dims for class/regr prediction')
    parser.add_argument('--prediction_latent_dim', type=int, default=0,
                help='prediction dims for class/regr prediction')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--do_chckpt", action="store_true",
                help="save model checkpoints")
    parser.add_argument("--predict_next", action="store_true", 
                help="use state_now to 'autoencode' state_next")
    parser.add_argument("--use_prev_input", action="store_true",
                help="use state_prev to help latent_now decode state_now")
    parser.add_argument('--patience', type=int, default=5,
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
                help="The type of data to fit ['piano', 'mnist']")
    parser.add_argument('--debug', action="store_true",
                help="if debug; then stop before model.fit")
    args = parser.parse_args()
    
    time_stamp = int(time())
    args.run_name = '{}_{}'.format(args.run_name, time_stamp)

    print('\n\n**\tRun Name: {}\t**\n\n'.format(args.run_name))

    if 'class' in args.network_type.lower():
        args.network_type = 'classification'
    if 'regr' in args.network_type.lower():
        args.network_type = 'regression'

    if args.network_type is 'regression': args.n_classes = 1

    data_types = ['piano', 'mnist', 'exoplanet']

    if 'piano' in args.data_type.lower():

        return_label_next = args.predict_next or args.use_prev_input

        P = PianoData(train_file = args.train_file,
                      batch_size = args.batch_size,
                      seq_length = args.seq_length,
                      step_length=args.step_length,
                      return_label_next = return_label_next,
                      squeeze_x = not args.no_squeeze_x,
                      squeeze_y = not args.no_squeeze_y)

        if args.seq_length > 1:
            X = vstack([P.data_train, 
                           P.data_valid, 
                           P.data_test, 
                           P.labels_train, 
                           P.labels_valid, 
                           P.labels_test
                         ])

            idx = X.sum(axis=0).sum(axis=0) > 0

            n_train = P.data_train.shape[0]
            n_valid = P.data_valid.shape[0]
            n_test = P.data_test.shape[0]

            P.data_train = P.data_train[:,:,idx].reshape((n_train, -1))
            P.data_valid = P.data_valid[:,:,idx].reshape((n_valid, -1))
            P.data_test = P.data_test[:,:,idx].reshape((n_test, -1))

            P.labels_train = P.labels_train[:,:,idx].reshape((n_train, -1))
            P.labels_valid = P.labels_valid[:,:,idx].reshape((n_valid, -1))
            P.labels_test = P.labels_test[:,:,idx].reshape((n_test, -1))

            args.original_dim = idx.sum()*args.seq_length
        
        data_instance = P

    elif 'mnist' in args.data_type.lower():
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        batch_size = args.batch_size
        
        n_samples_test = x_test.shape[0]
        n_samples_test = (n_samples_test // batch_size)*batch_size
        
        x_test = x_test[:n_samples_test]
        y_test = y_test[:n_samples_test]

        n_samples_train = x_train.shape[0]
        n_samples_train = (n_samples_train // batch_size)*batch_size
        
        x_train = x_train[:n_samples_train]
        y_train = y_train[:n_samples_train]

        image_size = x_train.shape[1]
        original_dim = image_size * image_size

        x_train = reshape(x_train, [-1, original_dim])
        x_test = reshape(x_test, [-1, original_dim])

        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        """These are all of the necessary `data_instance` components"""
        data_instance = BlankClass()

        data_instance.train_classes = y_train
        data_instance.valid_classes = y_test
        data_instance.test_classes = arange(0) # irrelevant(?)

        data_instance.data_train = x_train
        data_instance.data_valid = x_test
        data_instance.data_test = arange(0) # irrelevant(?)

        data_instance.labels_train = data_instance.data_train
        data_instance.labels_valid = data_instance.data_valid
    elif 'exoplanet' in args.data_type.lower():
        assert(os.path.exists(args.train_file))

        exoplanet_filename = 'exoplanet_spectral_database.joblib.save'
        # exoplanet_features = 'exoplanet_spectral_features_labels.joblib.save'
        # exoplanet_spectra = 'exoplanet_spectral_grid.joblib.save'

        exoplanet_filename = '{}/{}'.format(args.train_file,exoplanet_filename)

        input_specdb = joblib.load(exoplanet_filename)
        # features, labels = joblib.load(exoplanet_features)
        # spectral_grid = joblib.dump(exoplanet_spectra)

        x_train, y_train_raw = input_specdb[0]
        x_test, y_test_raw = input_specdb[1]
        y_train, y_test = input_specdb[2]

        """These are all of the necessary `data_instance` components"""
        data_instance = BlankClass()

        # these are our "labels"; the regresser will be conditioning on these
        data_instance.train_classes = x_train
        data_instance.valid_classes = x_test
        data_instance.test_classes = arange(0) # irrelevant(?)

        # these are our "features"; the VAE will be reproducing these
        data_instance.data_train = y_train
        data_instance.data_valid = y_test
        data_instance.data_test = arange(0) # irrelevant(?)

        # This is a 'copy' because the output must equal the input
        data_instance.labels_train = data_instance.data_train
        data_instance.labels_valid = data_instance.data_valid
    else:
        raise ValueError("`data_type` must be in list {}".format(data_types))

    n_train, n_features = data_instance.data_train.shape
    n_test, n_features = data_instance.data_valid.shape

    if args.original_dim is 0: args.original_dim = n_features
    
    vae_model, best_loss, history = train_vae_classifier(clargs = args, 
                                                data_instance = data_instance)

    print('\n\n[INFO] The Best Loss: {}'.format(best_loss))
    save_loc = '{}/{}_trained_model_output.joblib.save'.format(args.model_dir,
                                                               args.run_name)

    joblib.dump({'model':vae_model, 'best_loss':best_loss, 'history':history},
                save_loc)