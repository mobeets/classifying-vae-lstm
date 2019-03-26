import json
import numpy as np
import scipy.stats

from functools import partial, update_wrapper

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Reshape, concatenate
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.losses import mean_squared_error

from ..utils.midi_utils import write_sample

try:
    # Python 2 
    range 
except: 
    # Python 3
   def range(tmp): return iter(range(tmp))

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
                    clf_kl_weight = 1.0, optimizer = 'adam-wn', 
                    use_prev_input = False, network_type = 'classification'):
        
        self.network_type = network_type
        self.original_dim = original_dim
        self.vae_hidden_dim, self.vae_latent_dim = vae_dims
        self.clf_hidden_dim, self.class_dim = classifier_dims
        self.clf_latent_dim = clf_latent_dim or self.class_dim - 1
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_prev_input = use_prev_input
    
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
    
    def classifier_kl_loss(self, labels, preds):
        vs = 1 - self.clf_log_var_prior + self.clf_log_var
        vs = vs - K.exp(self.clf_log_var)/K.exp(self.clf_log_var_prior)
        vs = vs - K.square(self.clf_mean)/K.exp(self.clf_log_var_prior)
        return -0.5*K.sum(vs, axis = -1)

    def classifier_rec_loss(self, labels, preds):
        if self.network_type is 'classification':
            rec_loss = categorical_crossentropy(labels, preds)
        if self.network_type is 'regression':
            rec_loss = mean_squared_error(labels, preds)
        return self.clf_latent_dim * rec_loss

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
        clf_norm = concatenate([clf_norm, padding], name='classifier_norm')

        return K.exp(clf_norm)/K.sum(K.exp(clf_norm), axis = -1)[:,None]

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
        # batch_shape = (self.original_dim,)
        self.input_layer = Input(batch_shape = batch_shape, name='input_layer')

        if use_prev_input or self.use_prev_input:
            self.prev_input_layer = Input(batch_shape = batch_shape, 
                                        name = 'previous_input_layer')

        self.build_classifier()
        
        print('self.input_layer',self.input_layer)
        print('self.clf_pred',self.clf_pred)
        self.input_w_pred = concatenate([self.input_layer, self.clf_pred], 
                                            axis = -1,
                                            name = 'input_layer_w_clf_pred')

        self.build_latent_encoder()

        if use_prev_input or self.use_prev_input:
            self.prev_w_vae_latent = concatenate(
                [self.prev_input_layer, self.vae_latent_layer], 
                axis = -1, name = 'prev_inp_w_vae_lat_layer')
        else:
            self.prev_w_vae_latent = self.vae_latent_layer
        
        self.pred_w_latent = concatenate([self.clf_pred, 
                                          self.prev_w_vae_latent], 
                                          axis = -1,
                                          name = 'clf_pred_w_prev_w_vae_lat')
        
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

    def vae_sampling(self, args):
        eps = K.random_normal(shape = (self.batch_size, self.vae_latent_dim), 
                                mean = 0., stddev = 1.0)
        
        return self.vae_latent_mean + K.exp(self.vae_latent_log_var/2) * eps

    
    def vae_loss(self, input_layer, vae_decoded_mean):
        if self.network_type is 'classification':
            inp_vae_loss = binary_crossentropy(input_layer,vae_decoded_mean)
        if self.network_type is 'regression':
            inp_vae_loss = mean_squared_error(input_layer,vae_decoded_mean)
        return self.original_dim * inp_vae_loss

    def vae_kl_loss(self, ztrue, zpred):
        Z_mean = self.vae_latent_args[:,:self.vae_latent_dim]
        Z_log_var = self.vae_latent_args[:,self.vae_latent_dim:]
        k_summer = 1 + Z_log_var - K.square(Z_mean) - K.exp(Z_log_var)
        return -0.5*K.sum(k_summer, axis = -1)

    def load_model(self, model_file):
        """
        there's a currently bug in the way keras loads models 
            from `.yaml` files that has to do with `Lambda` calls
            ... so this is a hack for now
        """
        self.get_model()
        self.model.load_weights(model_file)

    def make_clf_encoder(self):
        batch_shape = (self.batch_size, self.original_dim)
        # batch_shape = (self.original_dim,)
        input_layer = Input(batch_shape = batch_shape, name = 'input_layer')

        # build label encoder
        enc_hidden_layer = self.model.get_layer('clf_hidden_layer')
        enc_hidden_layer = enc_hidden_layer(input_layer)
        
        clf_mean = self.model.get_layer('clf_mean')(enc_hidden_layer)
        clf_log_var = self.model.get_layer('clf_log_var')(enc_hidden_layer)

        return Model(input_layer, [clf_mean, clf_log_var])

    def make_latent_encoder(self):
        orig_batch_shape = (self.batch_size, self.original_dim)
        class_batch_shape = (self.batch_size, self.class_dim)
        # orig_batch_shape = (self.original_dim,)
        # class_batch_shape = (self.class_dim,)

        input_layer = Input(batch_shape = orig_batch_shape, 
                            name = 'input_layer')
        clf_layer = Input(batch_shape = class_batch_shape, 
                            name = 'classifier_layer')

        input_w_pred = concatenate([input_layer, clf_layer], axis = -1,
                                    name = 'input_w_clf_layer')
        
        # build latent encoder
        if self.vae_hidden_dim > 0:
            vae_hidden_layer = self.model.get_layer('vae_hidden_layer')
            vae_hidden_layer = vae_hidden_layer(input_w_pred)

            vae_latent_mean = self.model.get_layer('vae_latent_mean')
            vae_latent_mean = vae_latent_mean(vae_hidden_layer)

            vae_latent_log_var = self.model.get_layer('vae_latent_log_var')
            vae_latent_log_var = vae_latent_log_var(vae_hidden_layer)
        else:
            vae_latent_mean = self.model.get_layer('vae_latent_mean')
            vae_latent_mean = vae_latent_mean(input_w_pred)
            vae_latent_log_var= self.model.get_layer('vae_latent_log_var')
            vae_latent_log_var = vae_latent_log_var(input_w_pred)
        
        latent_enc_input = [input_layer, clf_layer]
        latent_enc_output = [vae_latent_mean, vae_latent_log_var]

        return Model(latent_enc_input, latent_enc_output)

    def make_latent_decoder(self, use_prev_input=False):

        input_batch_shape = (self.batch_size, self.original_dim)
        clf_batch_shape = (self.batch_size, self.class_dim)
        vae_batch_shape = (self.batch_size, self.vae_latent_dim)
        # input_batch_shape = (self.original_dim,)
        # clf_batch_shape = (self.class_dim,)
        # vae_batch_shape = (self.vae_latent_dim,)

        clf_layer = Input(batch_shape=clf_batch_shape, name='classifier_layer')
        vae_latent_layer = Input(batch_shape = vae_batch_shape, 
                                    name = 'vae_latent_layer')

        if use_prev_input or self.use_prev_input:
            prev_input_layer = Input(batch_shape = input_batch_shape, 
                                        name = 'prev_input_layer')

        if use_prev_input or self.use_prev_input:
            prev_vae_stack = [prev_input_layer, vae_latent_layer]
            prev_w_vae_latent = concatenate(prev_vae_stack, axis = -1)
        else:
            prev_w_vae_latent = vae_latent_layer

        pred_w_latent = concatenate([clf_layer, prev_w_vae_latent], axis = -1)

        # build physical decoder
        vae_decoded_mean = self.model.get_layer('vae_decoded_mean')
        if self.vae_hidden_dim > 0:
            vae_dec_hid_layer = self.model.get_layer('vae_dec_hid_layer')
            vae_dec_hid_layer = vae_dec_hid_layer(pred_w_latent)
            vae_decoded_mean = vae_decoded_mean(vae_dec_hid_layer)
        else:
            vae_decoded_mean = vae_decoded_mean(pred_w_latent)

        if use_prev_input or self.use_prev_input:
            dec_input_stack = [clf_layer, vae_latent_layer, prev_input_layer]
        else:
            dec_input_stack = [clf_layer, vae_latent_layer]

        return Model(dec_input_stack, vae_decoded_mean)

    def sample_classification(self, clf_mean, clf_log_var, nsamps=1, nrm_samp=False, add_noise=True):
        
        if nsamps == 1:
            eps = np.random.randn(*((1, clf_mean.flatten().shape[0])))
        else:
            eps = np.random.randn(*((nsamps,) + clf_mean.shape))
        if eps.T.shape == clf_mean.shape:
            eps = eps.T
        if add_noise:
            clf_norm = clf_mean + np.exp(clf_log_var/2)*eps
        else:
            clf_norm = self.clf_mean + 0*np.exp(self.clf_log_var/2)*eps

        if nrm_samp: return clf_norm

        if nsamps == 1:
            clf_norm = np.hstack([clf_norm, np.zeros((clf_norm.shape[0], 1))])
            output = np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis = -1)
            return output[:,None]
        else:
            clf_norm = np.dstack([clf_norm,np.zeros(clf_norm.shape[:-1]+(1,))])
            output = np.exp(clf_norm)/np.sum(np.exp(clf_norm), axis = -1)
            return output[:,:,None]

    def sample_latent(self, Z_mean, Z_log_var, nsamps = 1):
        if nsamps == 1:
            eps = np.random.randn(*Z_mean.squeeze().shape)
        else:
            eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
        return Z_mean + np.exp(Z_log_var/2) * eps

    def sample_vae(self, clf_mean):
        rando = np.random.rand(len(clf_mean.squeeze()))

        return np.float32(rando <= clf_mean)
