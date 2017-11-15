"""
Classifying VAE+LSTM (STORN)
"""
import argparse
import numpy as np
from keras import backend as K
from keras.utils import to_categorical
from utils.pianoroll import PianoData
from utils.model_utils import get_callbacks, save_model_in_pieces, AnnealLossWeight, init_adam_wn
from utils.weightnorm import data_based_init
from model import get_model

def train(args):
    P = PianoData(args.train_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        step_length=1,
        return_y_next=args.predict_next or args.use_x_prev,
        return_y_hist=True,
        squeeze_x=False,
        squeeze_y=False)    
    
    args.n_classes = len(np.unique(P.train_song_keys))
    w = to_categorical(P.train_song_keys, args.n_classes)
    wv = to_categorical(P.valid_song_keys, args.n_classes)

    print "Training with {} classes.".format(args.n_classes)
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
    model, _ = get_model(args.batch_size, args.original_dim, args.intermediate_dim, args.latent_dim, args.seq_length, args.n_classes, args.use_x_prev, args.optimizer, args.class_weight, kl_weight, w_kl_weight=w_kl_weight, w_log_var_prior=args.w_log_var_prior)
    args.optimizer = 'adam-wn' if was_adam_wn else args.optimizer
    save_model_in_pieces(model, args)

    print (P.x_train.shape, P.y_train.shape)
    if args.use_x_prev:
        x,y = [P.y_train, P.x_train], P.y_train
        xv,yv = [P.y_valid, P.x_valid], P.y_valid
        xt,yt = [P.y_test, P.x_test], P.y_test
    else:
        x,y = P.x_train, P.y_train
        xv,yv = P.x_valid, P.y_valid
        xt,yt = P.x_test, P.y_test
    xtr = x
    xva = xv
    xte = xt
    ytr = [y, w, w, y]
    yva = [yv, wv, wv, yv]

    data_based_init(model, x[:100])
    history = model.fit(xtr, ytr,
            shuffle=True,
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            validation_data=(xva, yva))
    best_ind = np.argmin([x if i >= min(args.kl_anneal, args.w_kl_anneal) else np.inf for i,x in enumerate(history.history['val_loss'])])
    best_loss = {k: history.history[k][best_ind] for k in history.history}
    return model, best_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
                help='tag for current run')
    parser.add_argument('--batch_size', type=int, default=200,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam-wn',
                help='optimizer name')
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--original_dim', type=int, default=88,
                help='input dim')
    parser.add_argument('--latent_dim', type=int, default=2,
                help='latent dim')
    parser.add_argument('--intermediate_dim', type=int, default=88,
                help='intermediate dim')
    parser.add_argument('--seq_length', type=int, default=16,
                help='sequence length (to use as history)')
    parser.add_argument('--class_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument("--predict_next", action="store_true", 
                help="use x_t to 'autoencode' x_{t+1}")
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--w_log_var_prior", type=float, default=0.0, 
                help="log variance prior on w")
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--patience', type=int, default=5,
                help='# of epochs, for early stopping')
    parser.add_argument("--use_x_prev", action="store_true",
                help="use x_{t-1} to help z_t decode x_t")
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
