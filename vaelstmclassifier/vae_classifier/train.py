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
        return_label_next=args.predict_next or args.use_prev_input,
        squeeze_x=True,
        squeeze_y=True)
    if args.seq_length > 1:
        X = np.vstack([P.data_train, P.data_valid, P.data_test, 
                        P.labels_train, P.labels_valid, P.labels_test])
        ix = X.sum(axis=0).sum(axis=0) > 0
        P.data_train = P.data_train[:,:,ix].reshape((len(P.data_train), -1))
        P.data_valid = P.data_valid[:,:,ix].reshape((len(P.data_valid), -1))
        P.data_test = P.data_test[:,:,ix].reshape((len(P.data_test), -1))
        P.labels_train = P.labels_train[:,:,ix].reshape((len(P.labels_train), -1))
        P.labels_valid = P.labels_valid[:,:,ix].reshape((len(P.labels_valid), -1))
        P.labels_test = P.labels_test[:,:,ix].reshape((len(P.labels_test), -1))
        args.original_dim = ix.sum()*args.seq_length

    args.n_classes = len(np.unique(P.train_classes))
    clf_train = to_categorical(P.train_classes, args.n_classes)
    clf_validation = to_categorical(P.valid_classes, args.n_classes)
    wte = to_categorical(P.test_classes, args.n_classes)

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
                         clf_kl_weight = clf_kl_weight)
    
    vae_clf.get_model()
    
    args.optimizer = 'adam-wn' if was_adam_wn else args.optimizer
    
    save_model_in_pieces(vae_clf.model, args)
    
    if args.use_prev_input:
        vae_train = [P.labels_train, P.data_train]
        vae_features_val = [P.labels_valid, P.data_valid]
    else:
        vae_train = P.data_train
        vae_features_val = P.data_valid

    data_based_init(vae_clf.model, P.data_train[:args.batch_size])

    vae_labels_val = [P.labels_valid,clf_validation, clf_validation,P.labels_valid]
    validation_data = (vae_features_val, vae_labels_val)
    train_labels = [P.labels_train, clf_train, clf_train, P.labels_train]
    
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