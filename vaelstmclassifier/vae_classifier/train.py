"""
    Classifying Variational Autoencoder
"""

import numpy as np

from keras import backend as K
from keras.utils import to_categorical
# from time import time

from ..utils.model_utils import get_callbacks, save_model_in_pieces
from ..utils.model_utils import init_adam_wn, AnnealLossWeight
from ..utils.weightnorm import data_based_init

from .model import VAEClassifier

def train_vae_classifier(clargs, data_instance, test_test = False):
    """Training control operations to create VAEClassifier instance, 
        organize the input data, and train the network.
    
    Args:
        clargs (object): command line arguments from `argparse`
            Structure Contents: 
                clargs.n_classes
                clargs.predict_next
                clargs.use_prev_input
                clargs.run_name
                clargs.patience
                clargs.kl_anneal
                clargs.do_log
                clargs.do_chkpt
                clargs.num_epochs
                clargs.w_kl_anneal
                clargs.optimizer
                clargs.batch_size

        data_instance (object): object instance for organizing data structures
            Structure Contents: 
                DI.train_classes
                DI.valid_classes
                DI.test_classes
                DI.labels_train
                DI.data_train
                DI.labels_valid
                DI.data_valid

        test_test (optional; bool): flag for storing classifier test parameters
    
    Returns:
        vae_clf (object): Variational AutoEncoder class instance
            Structure content: all data, all layers, training output, methods

        best_loss (dict): the best validation loss achieved during training
        
        history (object): output of the `keras` training procedures 
            Structure Contents:
                history (dict): loss, val_lss, etc
                epochs (list): list(range(num_epochs))
    """
    DI = data_instance

    clargs.n_classes = len(np.unique(DI.train_classes))
    clf_train = to_categorical(DI.train_classes, clargs.n_classes)
    clf_validation = to_categorical(DI.valid_classes, clargs.n_classes)

    if test_test: clf_test = to_categorical(DI.test_classes, clargs.n_classes)

    assert(not (clargs.predict_next and clargs.use_prev_input)), \
            "Can't use --predict_next if using --use_prev_input"

    # clargs.run_name = clargs.run_name + str(int(time()))
    callbacks = get_callbacks(clargs, patience=clargs.patience, 
                    min_epoch = max(clargs.kl_anneal, clargs.w_kl_anneal)+1, 
                    do_log = clargs.do_log, do_chckpt = clargs.do_chckpt)

    if clargs.kl_anneal > 0:
        assert(clargs.kl_anneal <= clargs.num_epochs), "invalid kl_anneal"
        vae_kl_weight = K.variable(value=0.1)
        callbacks += [AnnealLossWeight(vae_kl_weight, name="vae_kl_weight", 
                                final_value=1.0, n_epochs=clargs.kl_anneal)]
    else:
        vae_kl_weight = 1.0
    if clargs.w_kl_anneal > 0:
        assert(clargs.w_kl_anneal <= clargs.num_epochs), "invalid w_kl_anneal"
        clf_kl_weight = K.variable(value=0.0)
        callbacks += [AnnealLossWeight(clf_kl_weight, name="clf_kl_weight", 
                                final_value=1.0, n_epochs=clargs.w_kl_anneal)]
    else:
        clf_kl_weight = 1.0

    clargs.optimizer, was_adam_wn = init_adam_wn(clargs.optimizer)

    vae_dims = (clargs.vae_hidden_dim, clargs.vae_latent_dim)
    classifier_dims = (clargs.prediction_hidden_dim, clargs.n_classes)
    
    vae_clf = VAEClassifier(network_type = clargs.network_type,
                            batch_size = clargs.batch_size, 
                            original_dim = clargs.original_dim, 
                            vae_dims = vae_dims,
                            classifier_dims = classifier_dims, 
                            optimizer = clargs.optimizer,
                            clf_weight = clargs.predictor_weight, 
                            vae_kl_weight = vae_kl_weight, 
                            use_prev_input = clargs.use_prev_input,
                            clf_kl_weight = clf_kl_weight)
    
    vae_clf.get_model()
    
    clargs.optimizer = 'adam-wn' if was_adam_wn else clargs.optimizer
    
    save_model_in_pieces(vae_clf.model, clargs)
    
    if clargs.use_prev_input:
        vae_train = [DI.labels_train, DI.data_train]
        vae_features_val = [DI.labels_valid, DI.data_valid]
    else:
        vae_train = DI.data_train
        vae_features_val = DI.data_valid

    data_based_init(vae_clf.model, DI.data_train[:clargs.batch_size])

    vae_labels_val = [DI.labels_valid, clf_validation, 
                        clf_validation,DI.labels_valid]
    validation_data = (vae_features_val, vae_labels_val)
    train_labels = [DI.labels_train, clf_train, clf_train, DI.labels_train]
    
    if clargs.debug: return 0,0,0

    history = vae_clf.model.fit(vae_train, train_labels,
                                shuffle = True,
                                epochs = clargs.num_epochs,
                                batch_size = clargs.batch_size,
                                callbacks = callbacks,
                                validation_data = validation_data)

    best_ind = np.argmin([x if i >= max(clargs.kl_anneal,clargs.w_kl_anneal)+1\
                else np.inf for i,x in enumerate(history.history['val_loss'])])
    
    best_loss = {k: history.history[k][best_ind] for k in history.history}
    
    return vae_clf, best_loss, history