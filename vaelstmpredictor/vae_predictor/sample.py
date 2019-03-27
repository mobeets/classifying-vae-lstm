import argparse
import numpy as np

from collections import namedtuple
from json import load as json_load
from keras.utils import to_categorical

from ..utils.midi_utils import write_sample

# from .model import load_model, generate_sample, make_decoder
# from .model import make_w_encoder, make_z_encoder, sample_z
from .model import VAEClassifier
"""
Code to load pianoroll data (.pickle)
"""
import numpy as np

def generate_sample(instance, vae_enc_model, dec_model, 
                    data_prev, clf_t, data_prev_t,
                    use_latent_prior = False, 
                    use_prev_input = False):
    
    vaelat_mean, vaelat_log_var = vae_enc_model.predict(
                                            [data_prev, clf_t[:,None].T])
    ''' If `use_latent_prior`, then set distribution to N(0,1); 
    i.e. a standard normal by setting mod to 0
    '''
    mod = int(not use_latent_prior)
    latent_t = instance.sample_latent(mod*vaelat_mean, mod*vaelat_log_var)
    
    if use_prev_input or instance.use_prev_input:
        zc = [clf_t[:,None].T, latent_t, data_prev_t]
    else:
        zc = [clf_t[:,None].T, latent_t]
    
    vae_dec_mean = dec_model.predict(zc)
    
    return instance.sample_vae(vae_dec_mean)

def generate_sequence(instance, data_seed, num_steps, 
        clf_val = None, use_latent_prior=False, do_reset = True, 
        clf_sample = False, use_prev_input = False):
    """
    for t = 1:num_steps
        1. encode data_seed -> clf_mean, clf_log_var
        2. sample clf_t ~ logit-N(clf_mean, exp(clf_log_var/2))
        3. encode data_seed, clf_t -> vae_latent_mean, vae_latent_log_var
        4. sample latent_t ~ N(vae_latent_mean, exp(vae_latent_log_var/2))
        3. decode clf_t, latent_t -> clf_mean
        4. sample data_t ~ Bern(clf_mean)
        5. update data_seed := data_t
    """

    data_s = np.zeros([num_steps, instance.original_dim])
    data_prev = np.expand_dims(data_seed, axis=0)
    data_prev_t = data_prev
    clf_enc_model = instance.make_clf_encoder()
    if clf_val is None:
        clf_mean, clf_log_var = clf_enc_model.predict(data_prev)
        clf_t = instance.sample_classification(clf_mean, add_noise=clf_sample)
    else:
        clf_t = clf_val

    vae_enc_model = instance.make_latent_encoder()
    dec_model = instance.make_latent_decoder()

    for t in range(num_steps):
        data_t = generate_sample(instance, vae_enc_model, dec_model, 
                                    data_prev, clf_t, data_prev_t, 
                                    use_latent_prior, use_prev_input)
        data_s[t] = data_t
        
        data_prev_t = data_prev
        data_prev = data_t

    return data_s

def make_sample(generate_sequence, vae_instance, data_instance, clargs):
    # generate and write sample
    seed_ind = np.random.choice(list(range(len(data_instance.data_test))))
    data_seed = data_instance.data_test[seed_ind][0]

    # seed_key_ind = data_instance.test_song_keys[seed_ind] # 
    seed_class_ind = data_instance.test_labels[seed_ind] # 
    clf_val = None if clargs.infer_w else to_categorical(seed_class_ind, 
                                                     vae_instance.class_dim)

    sample = generate_sequence(vae_instance, data_seed, clargs.t, 
                                clf_val = clf_val, 
                                use_latent_prior = clargs.use_latent_prior)

    print('[INFO] Storing New MIDI file in {}/{}.mid'.format(
                                clargs.sample_dir, clargs.run_name))

    write_sample(sample, clargs.sample_dir, clargs.run_name, True)

def sample(clargs, data_instance):
    """Training control operations to create VAEClassifier instance, 
        organize the input data, and train the network.
    
    Args:
        clargs (object): command line arguments from `argparse`
            Structure Contents: 
                clargs.model_file
                clargs.num_samples
                clargs.infer_w
                clargs.t
                clargs.use_latent_prior
                clargs.sample_dir
                # clargs.n_labels
                # clargs.predict_next
                # clargs.use_prev_input
                # clargs.run_name
                # clargs.patience
                # clargs.kl_anneal
                # clargs.do_log
                # clargs.do_chkpt
                # clargs.num_epochs
                # clargs.w_kl_anneal
                # clargs.optimizer
                # clargs.batch_size

        data_instance (object): object instance for organizing data structures
            Structure Contents: 
                data_instance.data_test
                data_instance.test_labels
                # data_instance.train_labels
                # data_instance.valid_labels
                # data_instance.test_labels
                # data_instance.labels_train
                # data_instance.data_train
                # data_instance.labels_valid
                # data_instance.data_valid

    Returns:
        None: write midi to file
    """

    json_input = json_load(open(clargs.model_file.replace('.h5', '.json')))
    
    json_converter = namedtuple('json_converter', json_input.keys())
    margs = json_converter(**json_input)

    vae_dims = (margs.intermediate_dim, margs.latent_dim)
    predictor_dims = (margs.intermediate_class_dim, margs.n_labels)
    
    vae_clf = VAEClassifier(batch_size = margs.batch_size,
                            original_dim = margs.original_dim, 
                            vae_dims = vae_dims,
                            predictor_dims = predictor_dims, 
                            optimizer = margs.optimizer.replace('-wn',''),
                            clf_weight = margs.clf_weight, 
                            use_prev_input = margs.use_prev_input)
    
    vae_clf.get_model()
    vae_clf.model.load_weights(clargs.model_file)
    
    basenm = clargs.run_name
    for i in range(clargs.num_samples):
        clargs.run_name = basenm + '_' + str(i)
        make_sample(generate_sequence, vae_clf, data_instance, clargs)
