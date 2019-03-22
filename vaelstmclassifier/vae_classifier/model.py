import json
import numpy as np
import scipy.stats

from functools import partial, update_wrapper

from keras.layers import Input, Dense, Lambda, Reshape, concatenate
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy

try:
    # Python 2 
    range 
except: 
    # Python 3
   def range(tmp): return iter(range(tmp))

'''
def sample_vae(clf_mean):
    return 1.0*(np.random.rand(len(clf_mean.squeeze())) <= clf_mean)

def sample_classification(args, nsamps = 1, nrm_samp = False, 
                                add_noise = True):
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
        return np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis = -1)[:,None]
    else:
        clf_norm = np.dstack([clf_norm, np.zeros(clf_norm.shape[:-1]+ (1,))])
        return np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis = -1)[:,:,None]

def sample_z(args, nsamps = 1):
    Z_mean, Z_log_var = args
    if nsamps == 1:
        eps = np.random.randn(*Z_mean.squeeze().shape)
    else:
        eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
    return Z_mean + np.exp(Z_log_var/2) * eps

def make_w_encoder(model, original_dim, batch_size = 1):
    input_layer = Input(batch_shape = (batch_size, original_dim), 
                            name = 'input_layer')

    # build label encoder
    enc_hidden_layer = model.get_layer('enc_hidden_layer')(input_layer)
    clf_mean = model.get_layer('clf_mean')(enc_hidden_layer)
    clf_log_var = model.get_layer('clf_log_var')(enc_hidden_layer)

    return Model(input_layer, [clf_mean, clf_log_var])

def make_z_encoder(model, original_dim, class_dim, vae_dims, batch_size = 1):
    
    vae_hidden_dim, vae_latent_dim = vae_dims

    input_layer = Input(batch_shape = (batch_size, original_dim), 
                        name = 'input_layer')
    clf_layer = Input(batch_shape = (batch_size, class_dim), 
                        name = 'classifier_layer')
    input_w_pred = concatenate([input_layer, clf_layer], axis = -1)

    # build latent encoder
    if vae_hidden_dim > 0:
        vae_hidden_layer = model.get_layer('vae_hidden_layer')(input_w_pred)
        vae_latent_mean = model.get_layer('vae_latent_mean')(vae_hidden_layer)
        vae_latent_log_var = model.get_layer('vae_latent_log_var')
        vae_latent_log_var = vae_latent_log_var(vae_hidden_layer)
    else:
        vae_latent_mean = model.get_layer('vae_latent_mean')(input_w_pred)
        vae_latent_log_var= model.get_layer('vae_latent_log_var')(input_w_pred)

    return Model([input_layer, clf_layer], [vae_latent_mean, 
                                                vae_latent_log_var])

def make_decoder(model, vae_dims, class_dim, original_dim = 88, 
                    use_prev_input = False, batch_size = 1):

    vae_hidden_dim, vae_latent_dim = vae_dims

    clf_layer = Input(batch_shape = (batch_size, class_dim), 
                        name = 'classifier_layer')
    vae_latent_layer = Input(batch_shape = (batch_size, vae_latent_dim), 
                            name = 'vae_latent_layer')
    if use_prev_input or self.use_prev_input:
        prev_input_layer = Input(batch_shape = (batch_size, original_dim), 
                                    name = 'history')
    if use_prev_input or self.use_prev_input:
        prev_w_vae_latent = concatenate([prev_input_layer, vae_latent_layer], 
                                            axis = -1)
    else:
        prev_w_vae_latent = vae_latent_layer
    pred_w_latent = concatenate([clf_layer, prev_w_vae_latent], axis = -1)

    # build physical decoder
    vae_decoded_mean = model.get_layer('vae_decoded_mean')
    if vae_hidden_dim > 0:
        vae_dec_hid_layer = model.get_layer('vae_dec_hid_layer')
        vae_dec_hid_layer = vae_dec_hid_layer(pred_w_latent)
        vae_decoded_mean = vae_decoded_mean(vae_dec_hid_layer)
    else:
        vae_decoded_mean = vae_decoded_mean(pred_w_latent)

    if use_prev_input or self.use_prev_input:
        return Model([clf_layer, vae_latent_layer, prev_input_layer], 
                                                        vae_decoded_mean)
    else:
        return Model([clf_layer, vae_latent_layer], vae_decoded_mean)
'''

'''HERE WHERE I STARTED'''
def wrapped_partial(func, *args, **kwargs):
    '''
        from: http://louistiao.me/posts/
        adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    '''
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

class VAEClassifier(object):
    def __init__(self, original_dim, vae_dims, classifier_dims, 
                    clf_latent_dim = None, batch_size = 128, 
                    vae_kl_weight = 1.0, clf_weight=1.0, 
                    clf_kl_weight = 1.0, 
                    optimizer = 'adam-wn', use_prev_input = False):

        self.original_dim = original_dim
        self.vae_hidden_dim, self.vae_latent_dim = vae_dims
        self.clf_hidden_dim, self.class_dim = classifier_dims
        self.clf_latent_dim = clf_latent_dim or self.class_dim - 1

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_prev_input = use_prev_input
    
    def classifier_sampling(self, args):
        """
            sample from a logit-normal with params clf_mean and clf_log_var
                (n.b. this is very similar to a logistic-normal distribution)
        """
        eps = K.random_normal(shape = (self.batch_size, self.clf_latent_dim), 
                                mean = 0., stddev = 1.0)

        clf_norm = self.clf_mean + K.exp(self.clf_log_var/2) * eps
        
        # need to add 0's so we can sum it all to 1
        padding = K.tf.zeros(self.batch_size, 1)[:,None]
        clf_norm = concatenate([clf_norm, padding])

        return K.exp(clf_norm)/K.sum(K.exp(clf_norm), axis = -1)[:,None]

    def build_classifier(self):
        clf_hidden_layer = Dense(self.clf_hidden_dim, 
                                    activation = self.hidden_activation, 
                                    name = 'clf_hidden_layer')

        self.clf_hidden_layer = clf_hidden_layer(self.input_layer)

        clf_mean = Dense(self.clf_latent_dim, name = 'clf_mean')
        self.clf_mean = clf_mean(self.clf_hidden_layer)

        clf_log_var = Dense(self.clf_latent_dim, name = 'clf_log_var')
        self.clf_log_var = clf_log_var(self.clf_hidden_layer)

        clf_pred = Lambda(self.classifier_sampling, 
                            name = 'classifier_prediction')

        self.clf_pred = clf_pred([self.clf_mean, self.clf_log_var])

    def vae_sampling(self, args):
        eps = K.random_normal(shape = (self.batch_size, self.vae_latent_dim), 
                                mean = 0., stddev = 1.0)
        
        return self.vae_latent_mean + K.exp(self.vae_latent_log_var/2) * eps

    def build_latent_encoder(self):
        if self.vae_hidden_dim > 0:
            vae_hidden_layer = Dense(self.vae_hidden_dim, 
                                     activation = self.hidden_activation, 
                                     name = 'vae_hidden_layer')
            self.vae_hidden_layer = vae_hidden_layer(self.input_w_pred)
            
            vae_latent_mean = Dense(self.vae_latent_dim,name='vae_latent_mean')
            self.vae_latent_mean = vae_latent_mean(self.vae_hidden_layer)
            
            vae_latent_log_var =Dense(self.vae_latent_dim,
                                      name = 'vae_latent_log_var')
            self.vae_latent_log_var = vae_latent_log_var(self.vae_hidden_layer)
        else:
            vae_latent_mean = Dense(self.vae_latent_dim,name='vae_latent_mean')
            self.vae_latent_mean = vae_latent_mean(self.input_w_pred)
            vae_latent_log_var = Dense(vae_latent_dim, 
                                        name = 'vae_latent_log_var')
            self.vae_latent_log_var = vae_latent_log_var(self.input_w_pred)

        vae_latent_layer = Lambda(self.vae_sampling, name = 'vae_latent_layer')
        self.vae_latent_layer = vae_latent_layer([self.vae_latent_mean, 
                                                  self.vae_latent_log_var])

        self.vae_latent_args = concatenate([self.vae_latent_mean, 
                                            self.vae_latent_log_var], 
                                            axis = -1, 
                                            name = 'vae_latent_args')
    def build_decoder(self):
        vae_decoded_mean = Dense(self.original_dim, 
                                 activation = self.output_activation, 
                                 name = 'vae_decoded_mean')
        if self.vae_hidden_dim > 0:
            vae_dec_hid_layer = Dense(self.vae_hidden_dim, 
                                      activation = self.hidden_activation, 
                                      name = 'vae_dec_hid_layer')

            self.vae_dec_hid_layer = vae_dec_hid_layer(self.pred_w_latent)
            self.vae_decoded_mean = vae_decoded_mean(self.vae_dec_hid_layer)
        else:
            self.vae_decoded_mean = vae_decoded_mean(self.pred_w_latent)

    def vae_loss(self, input_layer, vae_decoded_mean):
        inp_vae_loss = binary_crossentropy(input_layer,vae_decoded_mean)
        return self.original_dim * inp_vae_loss

    def vae_kl_loss(self, ztrue, zpred):
        Z_mean = self.vae_latent_args[:,:self.vae_latent_dim]
        Z_log_var = self.vae_latent_args[:,self.vae_latent_dim:]
        k_summer = 1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var)
        return -0.5*K.sum(k_summer, axis = -1)

    def classifier_rec_loss(self, labels, preds):
        rec_loss = categorical_crossentropy(labels, preds)
        return self.clf_latent_dim * rec_loss

    def classifier_kl_loss(self, labels, preds):
        vs = 1 - self.clf_log_var_prior + self.clf_log_var
        vs = vs - K.exp(self.clf_log_var)/K.exp(self.clf_log_var_prior)
        vs = vs - K.square(self.clf_mean)/K.exp(self.clf_log_var_prior)
        return -0.5*K.sum(vs, axis = -1)

    def get_model(self, batch_size = None, original_dim = None, 
                  vae_dims = None, classifier_dims = None, 
                  clf_latent_dim = None, clf_weight = 1.0, vae_kl_weight = 1.0,
                  use_prev_input = False, clf_kl_weight = 1.0, 
                  clf_log_var_prior = 0.0, hidden_activation = 'relu', 
                  output_activation = 'sigmoid'):
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.clf_log_var_prior = clf_log_var_prior

        if vae_dims is not None:
            self.vae_hidden_dim, self.vae_latent_dim = vae_dims

        if classifier_dims is not None:
            self.clf_hidden_dim, self.class_dim = classifier_dims
        
        '''FINDME: Why is this class_dim-1(??)'''
        if clf_latent_dim is not None:
            self.clf_latent_dim = clf_latent_dim

        batch_shape = (self.batch_size, self.original_dim)
        self.input_layer = Input(batch_shape = batch_shape, name='input_layer')

        if use_prev_input or self.use_prev_input:
            self.prev_input_layer = Input(batch_shape = batch_shape, 
                                        name = 'previous_input_layer')

        self.build_classifier()
        
        self.input_w_pred = concatenate([self.input_layer, self.clf_pred], 
                                            axis = -1)

        self.build_latent_encoder()

        if use_prev_input or self.use_prev_input:
            self.prev_w_vae_latent = concatenate([self.prev_input_layer, 
                                                  self.vae_latent_layer], 
                                                  axis = -1)
        else:
            self.prev_w_vae_latent = self.vae_latent_layer
        
        self.pred_w_latent = concatenate([self.clf_pred, 
                                          self.prev_w_vae_latent], 
                                          axis = -1)
        
        self.build_decoder()

        # Add some wiggle to the classifier predictions 
        #   to avoid division by zero
        clf_pred_mod = Lambda(lambda tmp: tmp+1e-10, 
                                name = 'classifier_prediction_mod')
        self.clf_pred_mod = clf_pred_mod(self.clf_pred)

        if use_prev_input or self.use_prev_input:
            input_stack = [self.input_layer, self.prev_input_layer]
            out_stack = [self.vae_decoded_mean, self.clf_pred, 
                         self.clf_pred_mod, self.vae_latent_args]
            enc_stack = [self.vae_latent_mean, self.clf_mean]
        else:
            input_stack = [self.input_layer]
            out_stack = [self.vae_decoded_mean, self.clf_pred, 
                         self.clf_pred_mod, self.vae_latent_args]
            enc_stack = [self.vae_latent_mean, self.clf_mean]

        self.model = Model(input_stack, out_stack)
        self.enc_model = Model(input_stack, enc_stack)

        self.model.compile(  
                optimizer = self.optimizer,

                loss = {'vae_decoded_mean': self.vae_loss,
                        'classifier_prediction': self.classifier_kl_loss,
                        'classifier_prediction_mod': self.classifier_rec_loss,
                        'vae_latent_args': self.vae_kl_loss},

                loss_weights = {'vae_decoded_mean': 1.0,
                                'classifier_prediction': clf_kl_weight,
                                'classifier_prediction_mod':clf_weight,
                                'vae_latent_args': vae_kl_weight},

                metrics = {'classifier_prediction': 'accuracy'})

        # if use_prev_input or self.use_prev_input:
        #     enc_input_stack = [self.input_layer, self.prev_input_layer]
        # else:
        #     enc_input_stack = [self.input_layer]
        # 
        # enc_stack = [self.vae_latent_mean, self.clf_mean]
        # self.enc_model = Model(input_stack, enc_stack)

    def load_model(self, model_file, batch_size = None, no_x_prev = False):
        """
        there's a currently bug in the way keras loads models 
            from `.yaml` files that has to do with `Lambda` calls
            ... so this is a hack for now
        """
        self.margs = json.load(open(model_file.replace('.h5', '.json')))
        
        if batch_size is None:
            self.batch_size = self.margs['batch_size']
        else:
            self.batch_size = batch_size

        if no_x_prev or 'use_prev_input' not in self.margs.keys():
            self.margs['use_prev_input'] = False

        self.use_prev_input = self.margs['use_prev_input']
        self.original_dim = self.margs['original_dim']

        self.vae_hidden_dim = self.margs['intermediate_dim']
        self.vae_latent_dim = self.margs['vae_latent_dim']

        self.clf_hidden_dim = self.margs['classifier_hidden_dim']
        self.class_dim = self.margs['num_classes']
        
        self.clf_weight = self.margs['clf_weight']

        self.clf_latent_dim = self.margs['clf_latent_dim'] or self.class_dim-1
        
        self.get_model()

        self.model.load_weights(model_file)

    def generate_sample(self, x_seed, nsteps, w_val = None, use_z_prior=False, 
                    do_reset = True, w_sample = False, use_prev_input = False):
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
            w_t = sample_classification(w_enc_model.predict(x_prev), 
                                            add_noise = w_sample)
        else:
            w_t = w_val
        for t in range(nsteps):
            vae_latent_mean, vae_latent_log_var = z_enc_model.predict(
                                                        [x_prev, w_t[:,None].T])
            if use_z_prior:
                z_t = sample_z((0*vae_latent_mean, 0*vae_latent_log_var))
            else:
                z_t = sample_z((vae_latent_mean, vae_latent_log_var))
            if use_prev_input or self.use_prev_input:
                zc = [w_t[:,None].T, z_t, x_prev_t]
            else:
                zc = [w_t[:,None].T, z_t]
            
            x_t = sample_vae(dec_model.predict(zc))
            
            Xs[t] = x_t
            x_prev_t = x_prev
            x_prev = x_t
        return Xs

    def make_sample(self, P, args):
        # generate and write sample
        seed_ind = np.random.choice(list(range(len(P.x_test))))
        x_seed = P.x_test[seed_ind][0]
        seed_key_ind = P.test_song_keys[seed_ind]
        w_val = None if args.infer_w else to_categorical(seed_key_ind, 
                                                         self.class_dim)

        sample = self.generate_sample(x_seed, args.t,  w_val=w_val, 
                                    use_z_prior=args.use_z_prior)

        print('[INFO] Storing New MIDI file in {}/{}.mid'.format(
                                    args.sample_dir, args.run_name))

        write_sample(sample, args.sample_dir, args.run_name, True)