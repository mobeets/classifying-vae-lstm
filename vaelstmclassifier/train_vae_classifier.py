import argparse

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from tqdm import tqdm

from vaelstmclassifier.utils.pianoroll import PianoData
from vaelstmclassifier.vae_classifier import train
vae_classifier_train = train.train_vae_classifier # rename

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--batch_size', type=int, default=128,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=0,
                help='input dim')
    parser.add_argument('--intermediate_dim', type=int, default=128,
                help='intermediate dim')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--seq_length', type=int, default=1,
                help='sequence length (concat)')
    parser.add_argument('--clf_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--clf_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument('--intermediate_class_dim', type=int, default=128,
                help='intermediate dims for classes')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--do_chckpt", action="store_true",
                help="save model checkpoints")
    parser.add_argument("--predict_next", action="store_true", 
                help="use x_t to 'autoencode' x_{t+1}")
    parser.add_argument("--use_prev_input", action="store_true",
                help="use x_{t-1} to help z_t decode x_t")
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
    parser.add_argument('-sl','--step_length', type=int, default=1,
                help="Length of the step for overlap in song(s)")
    parser.add_argument('--data_type', type=str, default='piano',
                help="The type of data to fit ['piano', 'mnist']")
    parser.add_argument('--debug', action="store_true",
                help="if debug; then stop before model.fit")
    args = parser.parse_args()
    
    data_types = ['piano', 'mnist']

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

        class BlankClass(object):
            def __init__(self):
                pass

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
        wavelengths = None
        waves_use = None

        if verbose: print("[INFO] Load data from harddrive.")

        spectral_filenames = glob(args.train_file + '/trans*')
        spectral_grid = {}

        for fname in tqdm(spectral_filenames):
            key = '_'.join(fname.split('/')[-1].split('_')[1:7])
            info_now = loadtxt(fname)
            if wavelengths is None: wavelengths = info_now[:,0]
            if waves_use is None: waves_use = wavelengths < 5.0
            spectral_grid[key] = info_now[:,1][waves_use]

        if verbose: print("[INFO] Assigning input values onto `labels` and `features`")

        n_waves = waves_use.sum()
        labels = zeros((len(spectral_filenames), n_waves))
        features = zeros((len(spectral_filenames), len(key.split('_'))))

        for k, (key,val) in enumerate(spectral_grid.items()): 
            labels[k] = val
            features[k] = array(key.split('_')).astype(float)

        if verbose: print("[INFO] Computing train test split over indices "
                            "with shuffling")

        test_size = 0.2
        idx_train, idx_test = train_test_split(arange(len(spectral_filenames)), 
                                                test_size=test_size)

        if verbose: print("[INFO] Assigning x_train, y_train, x_test, y_test "
                            "from `idx_train` and `idx_test`")

        ''' Organize input data for autoencoder '''
        x_train = features[idx_train]
        y_train = labels[idx_train]

        x_test = features[idx_test]
        y_test = labels[idx_test]

        if verbose: print('Computing Median Spectrum')
        # y_train_med = median(y_train, axis=0)

        if verbose: print('Computing Median Average Deviation Spectrum')
        # y_train_mad = scale.mad(y_train, axis=0)
        min_train_mad = 1e-6

        y_train_max = y_train.max(axis=0)
        y_train_min = y_train.min(axis=0)

        y_train_range = (y_train_max - y_train_min + min_train_mad)
        
        y_train_fit = (y_train - y_train_min) / y_train_range
        y_test_fit = (y_test - y_train_min) / y_train_range

    else:
        raise ValueError("`data_type` must be in list {}".format(data_types))

    n_train, n_features = data_instance.data_train.shape
    n_test, n_features = data_instance.data_valid.shape

    if args.original_dim is 0: args.original_dim = n_features
    
    vae_clf, best_loss, history = vae_classifier_train(clargs = args, 
                                                data_instance = data_instance)

    print('\n\n[INFO] The Best Loss: {}'.format(best_loss))
