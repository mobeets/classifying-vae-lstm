"""
Classifying variational autoencoders
"""
import argparse
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from utils.pianoroll import PianoData
from utils.model_utils import get_callbacks, save_model_in_pieces, init_adam_wn, AnnealLossWeight
from utils.weightnorm import data_based_init
from model import get_model

def train(args):
    P = PianoData(args.train_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        step_length=1,
        return_y_next=args.predict_next or args.use_x_prev,
        squeeze_x=True,
        squeeze_y=True)
    if args.seq_length > 1:
        X = np.vstack([P.x_train, P.x_valid, P.x_test, P.y_train, P.y_valid, P.y_test])
        ix = X.sum(axis=0).sum(axis=0) > 0
        P.x_train = P.x_train[:,:,ix].reshape((len(P.x_train), -1))
        P.x_valid = P.x_valid[:,:,ix].reshape((len(P.x_valid), -1))
        P.x_test = P.x_test[:,:,ix].reshape((len(P.x_test), -1))
        P.y_train = P.y_train[:,:,ix].reshape((len(P.y_train), -1))
        P.y_valid = P.y_valid[:,:,ix].reshape((len(P.y_valid), -1))
        P.y_test = P.y_test[:,:,ix].reshape((len(P.y_test), -1))
        args.original_dim = ix.sum()*args.seq_length

    args.n_classes = len(np.unique(P.train_song_keys))
    wtr = to_categorical(P.train_song_keys, args.n_classes)
    wva = to_categorical(P.valid_song_keys, args.n_classes)
    wte = to_categorical(P.test_song_keys, args.n_classes)

    assert not (args.predict_next and args.use_x_prev), "Can't use --predict_next if using --use_x_prev"
    callbacks = get_callbacks(args, patience=args.patience, 
        min_epoch=max(args.kl_anneal, args.w_kl_anneal)+1, do_log=args.do_log)
    if args.kl_anneal > 0:
        assert args.kl_anneal <= args.num_epochs, "invalid kl_anneal"
        kl_weight = K.variable(value=0.1)
        callbacks += [AnnealLossWeight(kl_weight, name="kl_weight", final_value=1.0, n_epochs=args.kl_anneal)]
    else:
        kl_weight = 1.0
    if args.w_kl_anneal > 0:
        assert args.w_kl_anneal <= args.num_epochs, "invalid w_kl_anneal"
        w_kl_weight = K.variable(value=0.0)
        callbacks += [AnnealLossWeight(w_kl_weight, name="w_kl_weight", final_value=1.0, n_epochs=args.w_kl_anneal)]
    else:
        w_kl_weight = 1.0

    args.optimizer, was_adam_wn = init_adam_wn(args.optimizer)
    model, enc_model = get_model(args.batch_size, args.original_dim, (args.intermediate_dim, args.latent_dim), (args.intermediate_class_dim, args.n_classes), args.optimizer, args.class_weight, kl_weight, use_x_prev=args.use_x_prev, w_kl_weight=w_kl_weight, w_log_var_prior=args.w_log_var_prior)
    args.optimizer = 'adam-wn' if was_adam_wn else args.optimizer
    save_model_in_pieces(model, args)

    if args.use_x_prev:
        xtr = [P.y_train, P.x_train]
        xva = [P.y_valid, P.x_valid]
    else:
        xtr = P.x_train
        xva = P.x_valid

    data_based_init(model, P.x_train[:100])
    history = model.fit(xtr, [P.y_train, wtr, wtr, P.y_train],
            shuffle=True,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            validation_data=(xva, [P.y_valid, wva, wva, P.y_valid]))
    best_ind = np.argmin([x if i >= max(args.kl_anneal, args.w_kl_anneal)+1 else np.inf for i,x in enumerate(history.history['val_loss'])])
    best_loss = {k: history.history[k][best_ind] for k in history.history}
    return model, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--batch_size', type=int, default=100,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name') # 'rmsprop'
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=88,
                help='input dim')
    parser.add_argument('--intermediate_dim', type=int, default=88,
                help='intermediate dim')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--seq_length', type=int, default=1,
                help='sequence length (concat)')
    parser.add_argument('--class_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--w_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument('--intermediate_class_dim',
                type=int, default=88,
                help='intermediate dims for classes')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--predict_next", action="store_true", 
                help="use x_t to 'autoencode' x_{t+1}")
    parser.add_argument("--use_x_prev", action="store_true",
                help="use x_{t-1} to help z_t decode x_t")
    parser.add_argument('--patience', type=int, default=5,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='../data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str,
                default='../data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str,
                default='../data/input/JSB Chorales_Cs.pickle',
                help='file of training data (.pickle)')
    args = parser.parse_args()
    train(args)
