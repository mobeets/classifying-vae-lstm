"""
    Classifying Variational Autoencoder
"""

import numpy as np

from keras import backend as K
from keras.utils import to_categorical
from time import time

from ..utils.pianoroll import PianoData
from ..utils.model_utils import get_callbacks, save_model_in_pieces
from ..utils.model_utils import init_adam_wn, AnnealLossWeight
from ..utils.weightnorm import data_based_init

from .model import VAEClassifier

def train(args):
    P = PianoData(args.train_file,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        step_length=1,
        return_y_next=args.predict_next or args.use_prev_input,
        squeeze_x=True,
        squeeze_y=True)
    if args.seq_length > 1:
        X = np.vstack([P.x_train, P.x_valid, P.x_test, 
                        P.y_train, P.y_valid, P.y_test])
        ix = X.sum(axis=0).sum(axis=0) > 0
        P.x_train = P.x_train[:,:,ix].reshape((len(P.x_train), -1))
        P.x_valid = P.x_valid[:,:,ix].reshape((len(P.x_valid), -1))
        P.x_test = P.x_test[:,:,ix].reshape((len(P.x_test), -1))
        P.y_train = P.y_train[:,:,ix].reshape((len(P.y_train), -1))
        P.y_valid = P.y_valid[:,:,ix].reshape((len(P.y_valid), -1))
        P.y_test = P.y_test[:,:,ix].reshape((len(P.y_test), -1))
        args.original_dim = ix.sum()*args.seq_length

    args.n_classes = len(np.unique(P.train_song_keys))
    clf_train = to_categorical(P.train_song_keys, args.n_classes)
    clf_validation = to_categorical(P.valid_song_keys, args.n_classes)
    wte = to_categorical(P.test_song_keys, args.n_classes)

    assert(not (args.predict_next and args.use_prev_input)), \
            "Can't use --predict_next if using --use_prev_input"

    args.run_name = args.run_name + str(int(time()))
    callbacks = get_callbacks(args, patience=args.patience, 
                    min_epoch = max(args.kl_anneal, args.w_kl_anneal)+1, 
                    do_log = args.do_log, do_chckpt = args.do_chckpt)
    if args.kl_anneal > 0:
        assert(args.kl_anneal <= args.num_epochs), "invalid kl_anneal"
        vae_kl_weight = K.variable(value=0.1)
        callbacks += [AnnealLossWeight(vae_kl_weight, name="vae_kl_weight", 
                                    final_value=1.0, n_epochs=args.kl_anneal)]
    else:
        vae_kl_weight = 1.0
    if args.w_kl_anneal > 0:
        assert(args.w_kl_anneal <= args.num_epochs), "invalid w_kl_anneal"
        clf_kl_weight = K.variable(value=0.0)
        callbacks += [AnnealLossWeight(clf_kl_weight, name="clf_kl_weight", 
                                final_value=1.0, n_epochs=args.w_kl_anneal)]
    else:
        clf_kl_weight = 1.0

    args.optimizer, was_adam_wn = init_adam_wn(args.optimizer)

    vae_dims = (args.intermediate_dim, args.latent_dim)
    classifier_dims = (args.intermediate_class_dim, args.n_classes)

    vae_clf = VAEClassifier(batch_size = args.batch_size, 
                         original_dim = args.original_dim, 
                         vae_dims = vae_dims,
                         classifier_dims = classifier_dims, 
                         optimizer = args.optimizer,
                         clf_weight = args.clf_weight, 
                         vae_kl_weight = vae_kl_weight, 
                         use_prev_input = args.use_prev_input,
                         clf_kl_weight = clf_kl_weight)#, 
                         # clf_log_var_prior = args.clf_log_var_prior)

    vae_clf.get_model()
    
    args.optimizer = 'adam-wn' if was_adam_wn else args.optimizer

    save_model_in_pieces(vae_clf.model, args)

    if args.use_prev_input:
        vae_train = [P.y_train, P.x_train]
        vae_features_val = [P.y_valid, P.x_valid]
    else:
        vae_train = P.x_train
        vae_features_val = P.x_valid

    data_based_init(vae_clf.model, P.x_train[:args.batch_size])

    vae_labels_val = [P.y_valid,clf_validation, clf_validation,P.y_valid]
    validation_data = (vae_features_val, vae_labels_val)
    train_labels = [P.y_train, clf_train, clf_train, P.y_train]

    history = vae_clf.model.fit(vae_train, train_labels,
                                shuffle = True,
                                epochs = args.num_epochs,
                                batch_size = args.batch_size,
                                callbacks = callbacks,
                                validation_data = validation_data)

    best_ind = np.argmin([x if i >= max(args.kl_anneal, args.w_kl_anneal)+1 \
        else np.inf for i,x in enumerate(history.history['val_loss'])])
    
    best_loss = {k: history.history[k][best_ind] for k in history.history}
    
    return vae_clf, best_loss, history