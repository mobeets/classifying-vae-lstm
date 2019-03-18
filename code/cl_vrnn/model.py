import json
import numpy as np
from keras import losses
from keras import backend as K
from keras import initializers
from keras.layers import Input, Dense, LSTM, TimeDistributed, Lambda, concatenate, RepeatVector, Flatten
from keras.models import Model

def generate_sample(dec_model, w_enc_model, z_enc_model, x_seed, nsteps, use_x_prev, w_val=None, do_reset=True, seq_length=None, w_sample=False, w_discrete=False):
    """
    for t = 1:nsteps
        1. encode x_seed -> z_mean, z_log_var
        2. sample z_t ~ N(z_mean, exp(z_log_var/2))
        3. decode z_t -> x_mean
            - note: may also use x_{t-1}, depending on the model
        4. sample x_t ~ Bern(x_mean)
        5. update x_seed := x_t

    NOTE: Looks like what's assumed by STORN
    """
    if do_reset:
        dec_model.reset_states()
        w_enc_model.reset_states()
        z_enc_model.reset_states()
    # may need to seed model for multiple iters
    original_dim = x_seed.shape[-1]
    nseedsteps = x_seed.shape[0] if len(x_seed.shape) > 1 else 0
    Xs = np.zeros([nsteps+nseedsteps, original_dim])
    Ws = np.tile(w_val, (len(Xs), 1))
    if nseedsteps == 0:
        x_prev = x_seed[None,None,:]

    # decode w in the seed, or use provided value
    if w_val is None:
        ntms = x_seed.shape[1]
        w_ts = []
        for i in np.arange(0, ntms, seq_length):
            xcs = x_seed[i:i+seq_length]
            if xcs.shape[0] == seq_length:
                w_ts.append(sample_w(w_enc_model.predict(xcs[None,:]), add_noise=w_sample))
        w_t = np.vstack(w_ts).mean(axis=0)[None,:]
        # w_t = sample_w(w_enc_model.predict(x_seed[None,:seq_length]), add_noise=w_sample)
        if w_discrete:
            w_t = sample_w_discrete(w_t[0])[None,:]
    else:
        w_t = w_val
    for t in xrange(nsteps+nseedsteps):
        if t < nseedsteps:
            x_prev = x_seed[t][None,None,:]
        z_t = sample_z(z_enc_model.predict([x_prev, w_t]))

        # use previous X for decoding, if model requires this
        if use_x_prev:
            z_t = [z_t, x_prev, w_t]
        else:
            z_t = [z_t, w_t]
        x_t = sample_x(dec_model.predict(z_t))
        x_prev = x_t
        Xs[t] = x_t
    return Xs[nseedsteps:]

def sample_x(x_mean):
    return 1.0*(np.random.rand(*x_mean.squeeze().shape) <= x_mean)

def sample_w_discrete(w):
    wn = np.zeros(w.shape)
    wn[np.random.choice(len(w), p=w/w.sum())] = 1.
    # wn[np.argmax(w)] = 1.
    return wn

def sample_w(args, nsamps=1, nrm_samp=False, add_noise=True):
    w_mean, w_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*((1, w_mean.flatten().shape[0])))
    else:
        eps = np.random.randn(*((nsamps,) + w_mean.shape))
    if add_noise:
        w_norm = w_mean + np.exp(w_log_var/2)*eps
    else:
        w_norm = w_mean + 0*eps
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

def make_w_encoder(model, original_dim, n_classes, seq_length=1, batch_size=1):
    x = Input(batch_shape=(batch_size, seq_length, original_dim), name='x')

    # build label encoder
    hW =  model.get_layer('hW')
    encoder_w_layer =  model.get_layer('Wargs')
    # Wargs = encoder_w_layer(hW(x))
    Wargs = encoder_w_layer(hW(Flatten()(x)))
    def get_w_mean(x):
        return x[:,:(n_classes-1)]
    def get_w_log_var(x):
        return x[:,(n_classes-1):]
    w_mean = Lambda(get_w_mean)(Wargs)
    w_log_var = Lambda(get_w_log_var)(Wargs)

    mdl = Model(x, [w_mean, w_log_var])
    return mdl

def make_z_encoder(model, original_dim, n_classes, latent_dims, seq_length=1, batch_size=1, stateful=True):

    latent_dim_0, latent_dim = latent_dims

    x = Input(batch_shape=(batch_size, seq_length, original_dim), name='x')
    w = Input(batch_shape=(batch_size, n_classes), name='w')
    xw = concatenate([x, RepeatVector(seq_length)(w)], axis=-1)

    # build latent encoder
    h = LSTM(latent_dim_0,
        # activation='relu',
        stateful=stateful,
        return_sequences=True, name='encoder_h')(xw)
    Zm = Dense(latent_dim, name='Z_mean_t')
    Zv = Dense(latent_dim, name='Z_log_var_t')
    z_mean = TimeDistributed(Zm, name='Z_mean')(h)
    z_log_var = TimeDistributed(Zv, name='Z_log_var')(h)
    zm = model.get_layer('Z_mean')
    zv = model.get_layer('Z_log_var')
    Zm.set_weights(zm.get_weights())
    Zv.set_weights(zv.get_weights())

    mdl = Model([x, w], [z_mean, z_log_var])
    return mdl

def make_decoder(model, original_dim, intermediate_dim, latent_dim, n_classes, use_x_prev, seq_length=1, batch_size=1, stateful=True):
    # build decoder
    Z = Input(batch_shape=(batch_size, seq_length, latent_dim), name='Z')
    if use_x_prev:
        Xp = Input(batch_shape=(batch_size, seq_length, original_dim), name='history')
        XpZ = concatenate([Xp, Z], axis=-1)
    else:
        XpZ = Z
    W = Input(batch_shape=(batch_size, n_classes), name='W')
    XpZ = concatenate([XpZ, RepeatVector(seq_length)(W)], axis=-1)

    decoder_h = LSTM(intermediate_dim,
        # activation='relu',
        return_sequences=True,
        stateful=stateful, name='decoder_h')(XpZ)
    X_mean_t = Dense(original_dim, activation='sigmoid', name='X_mean_t')
    X_decoded_mean = TimeDistributed(X_mean_t, name='X_decoded_mean')(decoder_h)

    if use_x_prev:
        decoder = Model([Z, Xp, W], X_decoded_mean)
    else:
        decoder = Model([Z, W], X_decoded_mean)
    decoder.get_layer('X_decoded_mean').set_weights(model.get_layer('X_decoded_mean').get_weights())
    decoder.get_layer('decoder_h').set_weights(model.get_layer('decoder_h').get_weights())
    return decoder

def get_model(batch_size, original_dim, intermediate_dim, latent_dim, seq_length, n_classes, use_x_prev, optimizer, class_weight=1.0, kl_weight=1.0, dropout=0.0, w_kl_weight=1.0, w_log_var_prior = 0.0):
    """
    if intermediate_dim == 0, uses the output of the lstms directly
        otherwise, adds dense layers
    """
    X = Input(batch_shape=(batch_size, seq_length, original_dim), name='current')
    if use_x_prev:
        Xp = Input(batch_shape=(batch_size, seq_length, original_dim), name='history')

    # Sample w ~ logitNormal before continuing...
    hW = Dense(original_dim, activation='relu', name='hW')(Flatten()(X))
    Wargs = Dense(2*(n_classes-1), name='Wargs')(hW)
    def get_w_mean(x):
        return x[:,:(n_classes-1)]
    def get_w_log_var(x):
        return x[:,(n_classes-1):]
    W_mean = Lambda(get_w_mean)(Wargs)
    W_log_var = Lambda(get_w_log_var)(Wargs)
    # sample latents, w
    def sampling_w(args):
        W_mean, W_log_var = args
        eps = K.random_normal(shape=(batch_size, (n_classes-1)), mean=0., stddev=1.0)
        W_samp = W_mean + K.exp(W_log_var/2) * eps
        W0 = concatenate([W_samp, K.zeros((batch_size,1))], axis=-1)
        num = K.exp(W0)
        denom = K.sum(num, axis=-1, keepdims=True)
        return num/denom
    W = Lambda(sampling_w, output_shape=(n_classes,), name='W')([W_mean, W_log_var])
    
    XW = concatenate([X, RepeatVector(seq_length)(W)], axis=-1)

    # build encoder
    encoder_h = LSTM(intermediate_dim,
        # activation='relu',
        dropout=dropout,
        return_sequences=True, name='encoder_h')(XW)
    Z_mean_t = Dense(latent_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        name='Z_mean_t')
    Z_log_var_t = Dense(latent_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1),
        name='Z_log_var_t')
    Z_mean = TimeDistributed(Z_mean_t, name='Z_mean')(encoder_h)
    Z_log_var = TimeDistributed(Z_log_var_t, name='Z_log_var')(encoder_h)

    # sample latents, z
    def sampling(args):
        Z_mean, Z_log_var = args
        eps = K.random_normal(shape=(batch_size, seq_length, latent_dim), mean=0., stddev=1.0)
        return Z_mean + K.exp(Z_log_var/2) * eps
    Z = Lambda(sampling, output_shape=(seq_length, latent_dim,))([Z_mean, Z_log_var])

    if use_x_prev:
        XpZ = concatenate([Xp, Z], axis=-1)
    else:
        XpZ = Z
    XpZ = concatenate([XpZ, RepeatVector(seq_length)(W)], axis=-1)

    # build decoder
    decoder_h = LSTM(intermediate_dim,
        # activation='relu',
        dropout=dropout,
        return_sequences=True, name='decoder_h')(XpZ)
    X_mean_t = Dense(original_dim,
        bias_initializer='zeros',
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1), 
        activation='sigmoid',
        name='X_mean_t')
    X_decoded_mean = TimeDistributed(X_mean_t, name='X_decoded_mean')(decoder_h)

    def kl_loss(z_true, z_args):
        Z_mean = z_args[:,:,:latent_dim]
        Z_log_var = z_args[:,:,latent_dim:]
        return -0.5*K.sum(1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var), axis=-1)

    def vae_loss(X, X_decoded_mean):
        return original_dim * losses.binary_crossentropy(X, X_decoded_mean)

    def w_rec_loss(w_true, w):
        return (n_classes-1) * losses.categorical_crossentropy(w_true, w)
    
    def w_kl_loss(w_true, w):
        # w_log_var_prior
        # return -0.5 * K.sum(1 + W_log_var - K.exp(W_log_var) - K.square(W_mean), axis=-1)
        # vs = 1 + W_log_var - K.exp(W_log_var) - K.square(W_mean)
        vs = 1 - w_log_var_prior + W_log_var - K.exp(W_log_var)/K.exp(w_log_var_prior) - K.square(W_mean)/K.exp(w_log_var_prior)
        return -0.5*K.sum(vs, axis=-1)

    # n.b. have to add very small amount to rename :(
    W2 = Lambda(lambda x: x+1e-10, name='W2')(W)
    Z_args = concatenate([Z_mean, Z_log_var], axis=-1, name='Z_args')
    if use_x_prev:
        model = Model([X, Xp], [X_decoded_mean, W, W2, Z_args])
    else:
        model = Model(X, [X_decoded_mean, W, W2, Z_args])
    model.compile(optimizer=optimizer,
        loss={'X_decoded_mean': vae_loss, 'W': w_kl_loss, 'W2': w_rec_loss, 'Z_args': kl_loss},
        loss_weights={'X_decoded_mean': 1.0, 'W': w_kl_weight, 'W2': class_weight, 'Z_args': kl_weight},
        metrics={'W': 'accuracy'})

    encoder = Model(X, [Z_mean, Z_log_var, W])
    return model, encoder

def load_model(model_file, batch_size=None, seq_length=None, optimizer='adam'):
    """
    there's a curently bug in the way keras loads models from .yaml
        that has to do with Lambdas
    so this is a hack for now...
    """
    margs = json.load(open(model_file.replace('.h5', '.json')))
    # model = model_from_yaml(open(args.model_file))
    optimizer = margs['optimizer'] if optimizer is None else optimizer
    batch_size = margs['batch_size'] if batch_size is None else batch_size
    seq_length = margs['seq_length'] if seq_length is None else seq_length
    model, enc_model = get_model(batch_size, margs['original_dim'], margs['intermediate_dim'], margs['latent_dim'], seq_length, margs['n_classes'], margs['use_x_prev'], optimizer, margs['class_weight'])
    model.load_weights(model_file)
    return model, enc_model, margs
