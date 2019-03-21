import json
import numpy as np
import scipy.stats
from keras.layers import Input, Dense, Lambda, Reshape, concatenate
from keras.models import Model
from keras import backend as K
from keras import losses

try:
    # Python 2 
    range 
except: 
    # Python 3
   def range(tmp): return iter(range(tmp))

def generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, nsteps, 
                    w_val=None, use_z_prior=False, do_reset=True, 
                    w_sample=False, use_prev_input=False):
    """
    for t = 1:nsteps
        1. encode x_seed -> clf_mean, clf_log_var
        2. sample w_t ~ logit-N(clf_mean, exp(clf_log_var/2))
        3. encode x_seed, w_t -> vae_latent_mean, vae_latent_log_var
        4. sample z_t ~ N(vae_latent_mean, exp(vae_latent_log_var/2))
        3. decode w_t, z_t -> clf_mean
        4. sample x_t ~ Bern(clf_mean)
        5. update x_seed := x_t
    """
    original_dim = x_seed.shape[0]
    Xs = np.zeros([nsteps, original_dim])
    x_prev = np.expand_dims(x_seed, axis=0)
    x_prev_t = x_prev
    if w_val is None:
        w_t = sample_w(w_enc_model.predict(x_prev), add_noise=w_sample)
    else:
        w_t = w_val
    for t in range(nsteps):
        vae_latent_mean, vae_latent_log_var = z_enc_model.predict([x_prev, w_t[:,None].T])
        if use_z_prior:
            z_t = sample_z((0*vae_latent_mean, 0*vae_latent_log_var))
        else:
            z_t = sample_z((vae_latent_mean, vae_latent_log_var))
        if use_prev_input:
            zc = [w_t[:,None].T, z_t, x_prev_t]
        else:
            zc = [w_t[:,None].T, z_t]
        
        x_t = sample_x(dec_model.predict(zc))
        
        Xs[t] = x_t
        x_prev_t = x_prev
        x_prev = x_t
    return Xs

def sample_x(clf_mean):
    return 1.0*(np.random.rand(len(clf_mean.squeeze())) <= clf_mean)

def sample_w(args, nsamps=1, nrm_samp=False, add_noise=True):
    clf_mean, clf_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*((1, clf_mean.flatten().shape[0])))
    else:
        eps = np.random.randn(*((nsamps,) + clf_mean.shape))
    if eps.T.shape == clf_mean.shape:
        eps = eps.T
    if add_noise:
        clf_norm = clf_mean + np.exp(clf_log_var/2)*eps
    else:
        clf_norm = clf_mean + 0*np.exp(clf_log_var/2)*eps
    if nrm_samp:
        return clf_norm
    if nsamps == 1:
        clf_norm = np.hstack([clf_norm, np.zeros((clf_norm.shape[0], 1))])
        return np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis=-1)[:,None]
    else:
        clf_norm = np.dstack([clf_norm, np.zeros(clf_norm.shape[:-1]+ (1,))])
        return np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis=-1)[:,:,None]

def sample_z(args, nsamps=1):
    Z_mean, Z_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*Z_mean.squeeze().shape)
    else:
        eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
    return Z_mean + np.exp(Z_log_var/2) * eps

def make_w_encoder(model, original_dim, batch_size=1):
    input_layer = Input(batch_shape = (batch_size, original_dim), 
                            name = 'input_layer')

    # build label encoder
    h_w = model.get_layer('h_w')(input_layer)
    clf_mean = model.get_layer('clf_mean')(h_w)
    clf_log_var = model.get_layer('clf_log_var')(h_w)

    mdl = Model(input_layer, [clf_mean, clf_log_var])
    return mdl

def make_z_encoder(model, original_dim, class_dim, vae_dims, batch_size=1):
    
    vae_hidden_dim, vae_latent_dim = vae_dims

    input_layer = Input(batch_shape=(batch_size, original_dim), 
                        name = 'input_layer')
    clf_layer = Input(batch_shape=(batch_size, class_dim), 
                        name='classifier_layer')
    input_w_pred = concatenate([input_layer, clf_layer], axis=-1)

    # build latent encoder
    if vae_hidden_dim > 0:
        vae_hidden_layer = model.get_layer('vae_hidden_layer')(input_w_pred)
        vae_latent_mean = model.get_layer('vae_latent_mean')(vae_hidden_layer)
        vae_latent_log_var = model.get_layer('vae_latent_log_var')(vae_hidden_layer)
    else:
        vae_latent_mean = model.get_layer('vae_latent_mean')(input_w_pred)
        vae_latent_log_var = model.get_layer('vae_latent_log_var')(input_w_pred)

    mdl = Model([input_layer, clf_layer], [vae_latent_mean, vae_latent_log_var])
    return mdl

def make_decoder(model, vae_dims, class_dim, original_dim=88, use_prev_input=False, batch_size=1):

    vae_hidden_dim, vae_latent_dim = vae_dims

    clf_layer = Input(batch_shape = (batch_size, class_dim), 
                        name = 'classifier_layer')
    vae_latent_layer = Input(batch_shape = (batch_size, vae_latent_dim), 
                            name = 'vae_latent_layer')
    if use_prev_input:
        prev_input_layer = Input(batch_shape = (batch_size, original_dim), 
                                    name = 'history')
    if use_prev_input:
        prev_w_vae_latent = concatenate([prev_input_layer, vae_latent_layer], 
                                            axis=-1)
    else:
        prev_w_vae_latent = vae_latent_layer
    pred_w_latent = concatenate([clf_layer, prev_w_vae_latent], axis=-1)

    # build physical decoder
    vae_decoded_mean = model.get_layer('vae_decoded_mean')
    if vae_hidden_dim > 0:
        vae_dec_hid_layer = model.get_layer('vae_dec_hid_layer')
        vae_dec_hid_layer = vae_dec_hid_layer(pred_w_latent)
        vae_decoded_mean = vae_decoded_mean(vae_dec_hid_layer)
    else:
        vae_decoded_mean = vae_decoded_mean(pred_w_latent)

    if use_prev_input:
        mdl = Model([clf_layer, vae_latent_layer, prev_input_layer], vae_decoded_mean)
    else:
        mdl = Model([clf_layer, vae_latent_layer], vae_decoded_mean)
    return mdl

def get_model(batch_size, original_dim, vae_dims, classifier_dims, optimizer, 
                clf_weight=1.0, kl_weight=1.0, use_prev_input=False,
                w_kl_weight=1.0, clf_log_var_prior=0.0, clf_latent_dim=None):
    
    vae_hidden_dim, vae_latent_dim = vae_dims
    clf_hidden_dim, class_dim = classifier_dims
    
    '''FINDME: Why is this class_dim-1(??)'''
    clf_latent_dim = clf_latent_dim or class_dim-1

    input_layer = Input(batch_shape=(batch_size, original_dim), 
                                name='input_layer')
    if use_prev_input:
        prev_input_layer = Input(batch_shape=(batch_size, original_dim), 
                                    name='previous_input_layer')

    # build classifier
    clf_hidden_layer = Dense(clf_hidden_dim, activation = 'relu', 
                                             name = 'clf_hidden_layer')
    clf_hidden_layer = clf_hidden_layer(input_layer)

    clf_mean = Dense(clf_latent_dim, name='clf_mean')(clf_hidden_layer)
    clf_log_var = Dense(clf_latent_dim, name='clf_log_var')(clf_hidden_layer)

    # sample label
    def classifier_sampling(args):
        """
        sample from a logit-normal with params clf_mean and clf_log_var
            (n.b. this is very similar to a logistic-normal distribution)
        """
        clf_mean, clf_log_var = args
        eps = K.random_normal(shape = (batch_size, clf_latent_dim), 
                                mean=0., stddev=1.0)

        clf_norm = clf_mean + K.exp(clf_log_var/2) * eps
        
        # need to add '0' so we can sum it all to 1
        clf_norm = concatenate([clf_norm, K.tf.zeros(batch_size, 1)[:,None]])
        return K.exp(clf_norm)/K.sum(K.exp(clf_norm), axis=-1)[:,None]
    
    clf_pred = Lambda(classifier_sampling, name='classifier_prediction')
    clf_pred = clf_pred([clf_mean, clf_log_var])

    # build latent encoder
    input_w_pred = concatenate([input_layer, clf_pred], axis=-1)
    if vae_hidden_dim > 0:
        vae_hidden_layer = Dense(vae_hidden_dim, 
                                    activation='relu', 
                                    name='vae_hidden_layer')
        vae_hidden_layer = vae_hidden_layer(input_w_pred)
        
        vae_latent_mean = Dense(vae_latent_dim, name='vae_latent_mean')(vae_hidden_layer)
        vae_latent_log_var = Dense(vae_latent_dim, name='vae_latent_log_var')
        vae_latent_log_var = vae_latent_log_var(vae_hidden_layer)
    else:
        vae_latent_mean = Dense(vae_latent_dim, name='vae_latent_mean')(input_w_pred)
        vae_latent_log_var = Dense(vae_latent_dim, name='vae_latent_log_var')(input_w_pred)

    # sample latents
    def sampling(args):
        vae_latent_mean, vae_latent_log_var = args
        eps = K.random_normal(shape = (batch_size, vae_latent_dim), 
                                mean = 0., stddev = 1.0)
        
        return vae_latent_mean + K.exp(vae_latent_log_var/2) * eps

    vae_latent_layer = Lambda(sampling, name='vae_latent_layer')
    vae_latent_layer = vae_latent_layer([vae_latent_mean, vae_latent_log_var])

    # build decoder
    if use_prev_input:
        prev_w_vae_latent = concatenate([prev_input_layer, vae_latent_layer], axis=-1)
    else:
        prev_w_vae_latent = vae_latent_layer
    
    pred_w_latent = concatenate([clf_pred, prev_w_vae_latent], axis=-1)
    
    vae_decoded_mean = Dense(original_dim, activation = 'sigmoid', 
                                           name = 'vae_decoded_mean')
    if vae_hidden_dim > 0:
        vae_dec_hid_layer = Dense(vae_hidden_dim, activation = 'relu', 
                                                  name = 'vae_dec_hid_layer')

        vae_dec_hid_layer = vae_dec_hid_layer(pred_w_latent)
        vae_decoded_mean = vae_decoded_mean(vae_dec_hid_layer)
    else:
        vae_decoded_mean = vae_decoded_mean(pred_w_latent)

    def vae_loss(input_layer, vae_decoded_mean):
        inp_vae_loss = losses.binary_crossentropy(input_layer,vae_decoded_mean)
        return original_dim*inp_vae_loss

    def kl_loss(z_true, vae_latent_args):
        Z_mean = vae_latent_args[:,:vae_latent_dim]
        Z_log_var = vae_latent_args[:,vae_latent_dim:]
        return -0.5*K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

    def classifier_rec_loss(labels, preds):
        return clf_latent_dim * losses.categorical_crossentropy(labels, preds)

    # clf_log_var_prior = 1.0
    def classifier_kl_loss(labels, preds):
        # clf_log_var_prior
        # return -0.5 * K.sum(1 + clf_log_var - K.square(clf_mean) - K.exp(clf_log_var), axis=-1)
        vs = 1 - clf_log_var_prior + clf_log_var - K.exp(clf_log_var)/K.exp(clf_log_var_prior) - K.square(clf_mean)/K.exp(clf_log_var_prior)
        return -0.5*K.sum(vs, axis=-1)

    w2 = Lambda(lambda tmp: tmp+1e-10, name='w2')(clf_pred)
    vae_latent_args = concatenate([vae_latent_mean, vae_latent_log_var], 
                                        axis=-1, name='vae_latent_args')

    if use_prev_input:
        model = Model([input_layer, prev_input_layer], 
                        [vae_decoded_mean, clf_pred, w2, vae_latent_args])
        enc_model = Model([input_layer, prev_input_layer], [vae_latent_mean, clf_mean])
    else:
        model = Model(input_layer, [vae_decoded_mean, clf_pred, w2, vae_latent_args])
        enc_model = Model(input_layer, [vae_latent_mean, clf_mean])
    model.compile(optimizer=optimizer,
        loss={'vae_decoded_mean': vae_loss, 
                'classifier_prediction': classifier_kl_loss, 
                'w2': classifier_rec_loss, 'vae_latent_args': kl_loss},
        loss_weights={'vae_decoded_mean': 1.0, 
                    'classifier_prediction': w_kl_weight, 
                        'w2': clf_weight, 'vae_latent_args': kl_weight},
        metrics={'classifier_prediction': 'accuracy'})
    if use_prev_input:
        enc_model = Model([input_layer, prev_input_layer], [vae_latent_mean, clf_mean])
    else:
        enc_model = Model(input_layer, [vae_latent_mean, clf_mean])
    return model, enc_model

def load_model(model_file, optimizer='adam', batch_size=1, no_x_prev=False):
    """
    there's a curently bug in the way keras loads models from .yaml
        that has to do with Lambdas
    so this is a hack for now...
    """
    margs = json.load(open(model_file.replace('.h5', '.json')))
    # model = model_from_yaml(open(args.model_file))
    batch_size = margs['batch_size'] if batch_size == None else batch_size
    if no_x_prev or 'use_prev_input' not in margs:
        margs['use_prev_input'] = False
    model, enc_model = get_model(batch_size, margs['original_dim'], (margs['intermediate_dim'], margs['vae_latent_dim']), (margs['intermediate_class_dim'], margs['n_classes']), optimizer, margs['clf_weight'], use_prev_input=margs['use_prev_input'])
    model.load_weights(model_file)
    return model, enc_model, margs