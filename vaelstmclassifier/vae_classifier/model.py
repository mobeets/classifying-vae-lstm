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
   def range(x): return iter(range(x))

def generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, nsteps, 
                    w_val=None, use_z_prior=False, do_reset=True, 
                    w_sample=False, use_x_prev=False):
    """
    for t = 1:nsteps
        1. encode x_seed -> w_mean, w_log_var
        2. sample w_t ~ logit-N(w_mean, exp(w_log_var/2))
        3. encode x_seed, w_t -> z_mean, z_log_var
        4. sample z_t ~ N(z_mean, exp(z_log_var/2))
        3. decode w_t, z_t -> x_mean
        4. sample x_t ~ Bern(x_mean)
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
        z_mean, z_log_var = z_enc_model.predict([x_prev, w_t[:,None].T])
        if use_z_prior:
            z_t = sample_z((0*z_mean, 0*z_log_var))
        else:
            z_t = sample_z((z_mean, z_log_var))
        if use_x_prev:
            zc = [w_t[:,None].T, z_t, x_prev_t]
        else:
            zc = [w_t[:,None].T, z_t]
        
        x_t = sample_x(dec_model.predict(zc))
        
        Xs[t] = x_t
        x_prev_t = x_prev
        x_prev = x_t
    return Xs

def sample_x(x_mean):
    return 1.0*(np.random.rand(len(x_mean.squeeze())) <= x_mean)

def sample_w(args, nsamps=1, nrm_samp=False, add_noise=True):
    w_mean, w_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*((1, w_mean.flatten().shape[0])))
    else:
        eps = np.random.randn(*((nsamps,) + w_mean.shape))
    if eps.T.shape == w_mean.shape:
        eps = eps.T
    if add_noise:
        w_norm = w_mean + np.exp(w_log_var/2)*eps
    else:
        w_norm = w_mean + 0*np.exp(w_log_var/2)*eps
    if nrm_samp:
        return w_norm
    if nsamps == 1:
        w_norm = np.hstack([w_norm, np.zeros((w_norm.shape[0], 1))])
        return np.exp(w_norm)/np.sum(np.exp(w_norm), axis=-1)[:,None]
    else:
        w_norm = np.dstack([w_norm, np.zeros(w_norm.shape[:-1]+ (1,))])
        return np.exp(w_norm)/np.sum(np.exp(w_norm), axis=-1)[:,:,None]

def sample_z(args, nsamps=1):
    Z_mean, Z_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*Z_mean.squeeze().shape)
    else:
        eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
    return Z_mean + np.exp(Z_log_var/2) * eps

def make_w_encoder(model, original_dim, batch_size=1):
    x = Input(batch_shape=(batch_size, original_dim), name='x')

    # build label encoder
    clf_hidden_layer = model.get_layer('clf_hidden_layer')(x)
    w_mean = model.get_layer('w_mean')(clf_hidden_layer)
    w_log_var = model.get_layer('w_log_var')(clf_hidden_layer)

    mdl = Model(x, [w_mean, w_log_var])
    return mdl

def make_z_encoder(model, original_dim, class_dim, latent_dims, batch_size=1):
    
    encoder_hidden_dim, latent_dim = latent_dims

    input_encoder = Input(batch_shape=(batch_size, original_dim), name='x')
    input_classifier = Input(batch_shape=(batch_size, class_dim), name='w')
    input_w_clf = concatenate([input_encoder, input_classifier], axis=-1)

    # build latent encoder
    if encoder_hidden_dim > 0:
        enc_hidden_layer = model.get_layer('h')(input_w_clf)
        latent_mean = model.get_layer('latent_mean')(enc_hidden_layer)
        latent_log_var = model.get_layer('latent_log_var')(enc_hidden_layer)
    else:
        latent_mean = model.get_layer('latent_mean')(input_w_clf)
        latent_log_var = model.get_layer('latent_log_var')(input_w_clf)

    return  Model([input_encoder, input_classifier], \
                    [latent_mean, latent_log_var])

def make_decoder(model, latent_dims, class_dim, original_dim=88, use_x_prev=False, batch_size=1):

    encoder_hidden_dim, latent_dim = latent_dims

    w = Input(batch_shape=(batch_size, class_dim), name='w')
    z = Input(batch_shape=(batch_size, latent_dim), name='z')
    if use_x_prev:
        xp = Input(batch_shape=(batch_size, original_dim), name='history')
    if use_x_prev:
        xpz = concatenate([xp, z], axis=-1)
    else:
        xpz = z
    wz = concatenate([w, xpz], axis=-1)

    # build x decoder
    decoder_mean = model.get_layer('x_decoded_mean')
    if encoder_hidden_dim > 0:
        decoder_h = model.get_layer('decoder_h')
        h_decoded = decoder_h(wz)
        x_decoded_mean = decoder_mean(h_decoded)
    else:
        x_decoded_mean = decoder_mean(wz)

    if use_x_prev:
        mdl = Model([w, z, xp], x_decoded_mean)
    else:
        mdl = Model([w, z], x_decoded_mean)
    return mdl

def get_model(batch_size, original_dim, vae_dims,
              classifier_dims, optimizer, clf_lat_dim = None, 
              class_weight = 1.0, clf_dec_weight = 1.0, vae_kl_weight = 1.0, 
              clf_kl_weight = 1.0, use_prev_input = False, 
              clf_log_var_prior = 0.0,
              clf_hid_activation = 'relu', enc_hid_activation = 'relu', 
              dec_hid_activation = 'relu', decoder_activation = 'sigmoid',
              wiggle_room = 1e-10):
    
    encoder_hidden_dim, latent_dim = vae_dims
    classifier_hidden_dim, class_dim = classifier_dims
    
    '''FINDME: Why was this sized `class_dim-1`??'''
    clf_lat_dim = clf_lat_dim or class_dim - 1

    input_layer = Input(batch_shape = (batch_size, original_dim), 
                        name = 'input_layer')
    if use_prev_input:
        prev_input_layer = Input(batch_shape = (batch_size, original_dim), 
                                    name = 'previous_input_layer')

    # build label encoder
    clf_hidden_layer = Dense(classifier_hidden_dim, 
                                activation = clf_hid_activation, 
                                name = 'classifier_hidden_layer')(input_layer)

    clf_mean = Dense(class_dim, name = 'classifier_mean')
    clf_log_var = Dense(class_dim, name = 'classifier_log_var')

    clf_mean = clf_mean(clf_hidden_layer)
    clf_log_var = clf_log_var(clf_hidden_layer)

    # sample label
    def classification_sampling(args):
        """
        sample from a logit-normal with params clf_mean and clf_log_var
            (n.b. this is very similar to a logistic-normal distribution)
        """
        clf_mean, clf_log_var = args
        
        eps = K.random_normal(shape = (batch_size, clf_lat_dim), 
                                mean = 0., stddev=1.0)

        clf_norm = clf_mean + K.exp(clf_log_var/2) * eps

        # need to add '0' so we can sum it all to 1
        clf_norm = concatenate([clf_norm, K.tf.zeros(batch_size, 1)[:,None]])
        return K.exp(clf_norm)/K.sum(K.exp(clf_norm), axis = -1)[:,None]

    classifier = Lambda(classification_sampling, name = 'classifier')
    classifier = classifier([clf_mean, clf_log_var])

    # build latent encoder
    input_w_clf = concatenate([input_layer, classifier], axis = -1)
    if encoder_hidden_dim > 0:
        enc_hidden_layer = Dense(encoder_hidden_dim, 
                                    activation = enc_hid_activation, 
                                    name = 'encoder_hidden_layer')(input_w_clf)

        latent_mean = Dense(latent_dim, name = 'vae_latent_mean')
        latent_log_var = Dense(latent_dim, name = 'vae_latent_log_var')

        latent_mean = latent_mean(enc_hidden_layer)
        latent_log_var = latent_log_var(enc_hidden_layer)

    else:
        latent_mean = Dense(latent_dim, name = 'vae_latent_mean')
        latent_log_var = Dense(latent_dim, name = 'vae_latent_log_var')

        latent_mean = latent_mean(input_w_clf)
        latent_log_var = latent_log_var(input_w_clf)

    # sample latents
    def encoder_sampling(args):
        latent_mean, latent_log_var = args
        eps = K.random_normal(shape = (batch_size, latent_dim), 
                                mean = 0., stddev = 1.0)
        return latent_mean + K.exp(latent_log_var/2) * eps
    
    latent_layer = Lambda(encoder_sampling, name = 'latent_layer')
    latent_layer = latent_layer([latent_mean, latent_log_var])

    # build decoder
    if use_prev_input:
        prev_w_lat_layer = concatenate([prev_input_layer,latent_layer],axis=-1)
    else:
        prev_w_lat_layer = latent_layer
    
    classifier_w_lat = concatenate([classifier, prev_w_lat_layer], axis = -1)
    
    decoder_mean = Dense(original_dim, 
                            activation = decoder_activation, 
                            name = 'clf_decoded_mean')

    if encoder_hidden_dim > 0:
        dec_hidden_layer = Dense(encoder_hidden_dim, 
                                    activation = dec_hid_activation, 
                                    name = 'dec_hidden_layer')

        dec_hidden_layer = dec_hidden_layer(classifier_w_lat)
        clf_dec_mean = decoder_mean(dec_hidden_layer)
    else:
        clf_dec_mean = decoder_mean(classifier_w_lat)

    def vae_loss(input_layer, decoded_mean):
        return original_dim*losses.binary_crossentropy(input_layer,
                                                        decoded_mean)

    def vae_kl_loss(z_true, latent_args):
        ''' The variable names here confuse me a little;
                but they are all temporary; so we can keep them.
            Need to investigate how keras.models.Model.compile 
                calls on  multiple manually configure loss terms'''
        Z_mean = latent_args[:,:latent_dim]
        Z_log_var = latent_args[:,latent_dim:]

        kl = 1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var)
        return -0.5*K.sum(kl, axis = -1)

    def clf_rec_loss(clf_true, classifier):
        return clf_lat_dim*losses.categorical_crossentropy(clf_true,classifier)

    # clf_log_var_prior = 1.0
    def clf_kl_loss(labels, predictions):
        vs = 1 - clf_log_var_prior + clf_log_var 
        vs = vs - K.exp(clf_log_var)/K.exp(clf_log_var_prior) 
        vs = vs - K.square(clf_mean)/K.exp(clf_log_var_prior)

        return -0.5*K.sum(vs, axis = -1)

    classifier_mod = Lambda(lambda tmp: tmp+wiggle_room, name='classifier_mod')
    classifier_mod = classifier_mod(classifier)

    latent_args = concatenate([latent_mean, latent_log_var], 
                                axis = -1, name = 'latent_args')
    
    if use_prev_input:
        input_stack = [input_layer, prev_input_layer]
    else:
        input_stack = [input_layer]
    
    lat_out_stack = [latent_mean, clf_mean]
    clf_out_stack = [clf_dec_mean, classifier, classifier_mod, latent_args]

    model = Model(input_stack, clf_out_stack)
    enc_model = Model(input_stack, [latent_mean, clf_mean])

    model.compile(optimizer = optimizer,
                  loss = {'clf_decoded_mean': vae_loss, 
                          'classifier': clf_kl_loss, 
                          'classifier_mod': clf_rec_loss, 
                          'latent_args': vae_kl_loss},
                  loss_weights = {'clf_decoded_mean': clf_dec_weight, 
                                  'classifier': clf_kl_weight, 
                                  'classifier_mod': class_weight, 
                                  'latent_args': vae_kl_weight},
                  metrics = {'classifier': 'accuracy'})

    lat_out_stack = [latent_mean, clf_mean]
    
    if use_prev_input:
        input_stack = [input_layer, prev_input_layer]
    else:
        input_stack = [input_layer] # may need to remove the brackets

    enc_model = Model(input_stack, lat_out_stack)

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
    if no_x_prev or 'use_x_prev' not in margs:
        margs['use_x_prev'] = False
    model, enc_model = get_model(batch_size, margs['original_dim'], (margs['intermediate_dim'], margs['latent_dim']), (margs['intermediate_class_dim'], margs['n_classes']), optimizer, margs['class_weight'], use_x_prev=margs['use_x_prev'])
    model.load_weights(model_file)
    return model, enc_model, margs
