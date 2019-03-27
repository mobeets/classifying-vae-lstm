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
    def __init__(self, original_dim, vae_dims, predictor_dims, 
                    prediction_latent_dim = None, batch_size = 128, 
                    vae_kl_weight = 1.0, predictor_weight=1.0, 
                    predictor_kl_weight = 1.0, optimizer = 'adam-wn', 
                    use_prev_input = False, predictor_type = 'classification'):

        self.predictor_type = predictor_type
        self.original_dim = original_dim
        self.vae_hidden_dim, self.vae_latent_dim = vae_dims
        self.predictor_hidden_dim, self.predictor_out_dim = predictor_dims
        self.prediction_latent_dim = prediction_latent_dim or self.predictor_out_dim - 1

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_prev_input = use_prev_input
    
    def build_predictor(self):
        predictor_hidden_layer = Dense(self.predictor_hidden_dim, 
                                    activation = self.hidden_activation, 
                                    name = 'predictor_hidden_layer')

        self.predictor_hidden_layer = predictor_hidden_layer(self.input_layer)
        
        predictor_latent_mean = Dense(self.prediction_latent_dim, name = 'predictor_latent_mean')
        self.predictor_latent_mean = predictor_latent_mean(self.predictor_hidden_layer)

        prediction_log_var = Dense(self.prediction_latent_dim, name = 'prediction_log_var')
        self.prediction_log_var = prediction_log_var(self.predictor_hidden_layer)

        prediction_latent_layer = Lambda(self.predictor_sampling, 
                            name = 'predictor_prediction')

        self.prediction_latent_layer = prediction_latent_layer([self.predictor_latent_mean, self.prediction_log_var])

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
    
    def predictor_kl_loss(self, labels, preds):
        vs = 1 - self.prediction_log_var_prior + self.prediction_log_var
        vs = vs - K.exp(self.prediction_log_var)/K.exp(self.prediction_log_var_prior)
        vs = vs - K.square(self.predictor_latent_mean)/K.exp(self.prediction_log_var_prior)
        return -0.5*K.sum(vs, axis = -1)

    def predictor_rec_loss(self, labels, preds):
        if self.predictor_type is 'classification':
            rec_loss = categorical_crossentropy(labels, preds)
        if self.predictor_type is 'regression':
            rec_loss = mean_squared_error(labels, preds)
        return self.prediction_latent_dim * rec_loss

    def predictor_sampling(self, args):
        """
            sample from a logit-normal with params predictor_latent_mean and prediction_log_var
                (n.b. this is very similar to a logistic-normal distribution)
        """
        eps = K.random_normal(shape = (self.batch_size, self.prediction_latent_dim), 
                                mean = 0., stddev = 1.0)

        predictor_norm = self.predictor_latent_mean + K.exp(self.prediction_log_var/2) * eps
        
        # need to add 0's so we can sum it all to 1
        padding = K.tf.zeros(self.batch_size, 1)[:,None]
        predictor_norm = concatenate([predictor_norm, padding], name='predictor_norm')

        return K.exp(predictor_norm)/K.sum(K.exp(predictor_norm), axis = -1)[:,None]

    def get_model(self, batch_size = None, original_dim = None, 
                  vae_dims = None, predictor_dims = None, 
                  prediction_latent_dim = None, predictor_weight = 1.0, 
                  vae_kl_weight = 1.0, use_prev_input = False, 
                  predictor_kl_weight = 1.0, prediction_log_var_prior = 0.0, 
                  hidden_activation = 'relu', output_activation = 'sigmoid'):
        
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.prediction_log_var_prior = prediction_log_var_prior

        if vae_dims is not None:
            self.vae_hidden_dim, self.vae_latent_dim = vae_dims

        if predictor_dims is not None:
            self.predictor_hidden_dim, self.predictor_out_dim = predictor_dims
        
        '''FINDME: Why is this predictor_out_dim-1(??)'''
        if prediction_latent_dim is not None:
            self.prediction_latent_dim = prediction_latent_dim

        batch_shape = (self.batch_size, self.original_dim)
        # batch_shape = (self.original_dim,)
        self.input_layer = Input(batch_shape = batch_shape, name='input_layer')

        if use_prev_input or self.use_prev_input:
            self.prev_input_layer = Input(batch_shape = batch_shape, 
                                        name = 'previous_input_layer')

        self.build_predictor()
        
        self.input_w_pred = concatenate([self.input_layer, self.prediction_latent_layer], 
                                            axis = -1,
                                            name = 'data_input_w_pred_latent_out')

        self.build_latent_encoder()

        if use_prev_input or self.use_prev_input:
            self.prev_w_vae_latent = concatenate(
                [self.prev_input_layer, self.vae_latent_layer], 
                axis = -1, name = 'prev_inp_w_vae_lat_layer')
        else:
            self.prev_w_vae_latent = self.vae_latent_layer
        
        self.pred_w_latent = concatenate([self.prediction_latent_layer, 
                                          self.prev_w_vae_latent], 
                                          axis = -1,
                                          name = 'pred_latent_out_w_prev_w_vae_lat')
        
        self.build_decoder()

        # Add some wiggle to the predictor predictions 
        #   to avoid division by zero
        prediction_latent_mod = Lambda(lambda tmp: tmp+1e-10, 
                                name = 'predictor_prediction_mod')

        self.prediction_latent_mod = prediction_latent_mod(self.prediction_latent_layer)
        
        if use_prev_input or self.use_prev_input:
            input_stack = [self.input_layer, self.prev_input_layer]
            out_stack = [self.vae_decoded_mean, self.prediction_latent_layer, 
                         self.prediction_latent_mod, self.vae_latent_args]
            enc_stack = [self.vae_latent_mean, self.predictor_latent_mean]
        else:
            input_stack = [self.input_layer]
            out_stack = [self.vae_decoded_mean, self.prediction_latent_layer, 
                         self.prediction_latent_mod, self.vae_latent_args]
            enc_stack = [self.vae_latent_mean, self.predictor_latent_mean]

        self.model = Model(input_stack, out_stack)
        self.enc_model = Model(input_stack, enc_stack)

        self.model.compile(  
                optimizer = self.optimizer,

                loss = {'vae_decoded_mean': self.vae_loss,
                        'predictor_prediction': self.predictor_kl_loss,
                        'predictor_prediction_mod': self.predictor_rec_loss,
                        'vae_latent_args': self.vae_kl_loss},

                loss_weights = {'vae_decoded_mean': 1.0,
                                'predictor_prediction': predictor_kl_weight,
                                'predictor_prediction_mod':predictor_weight,
                                'vae_latent_args': vae_kl_weight},

                metrics = {'predictor_prediction': 'accuracy'})

    def vae_sampling(self, args):
        eps = K.random_normal(shape = (self.batch_size, self.vae_latent_dim), 
                                mean = 0., stddev = 1.0)
        
        return self.vae_latent_mean + K.exp(self.vae_latent_log_var/2) * eps

    
    def vae_loss(self, input_layer, vae_decoded_mean):
        if self.predictor_type is 'classification':
            inp_vae_loss = binary_crossentropy(input_layer,vae_decoded_mean)
        if self.predictor_type is 'regression':
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

    def make_predictor_encoder(self):
        batch_shape = (self.batch_size, self.original_dim)
        # batch_shape = (self.original_dim,)
        input_layer = Input(batch_shape = batch_shape, name = 'input_layer')

        # build label encoder
        enc_hidden_layer = self.model.get_layer('predictor_hidden_layer')
        enc_hidden_layer = enc_hidden_layer(input_layer)
        
        predictor_latent_mean = self.model.get_layer('predictor_latent_mean')(enc_hidden_layer)
        prediction_log_var = self.model.get_layer('prediction_log_var')(enc_hidden_layer)

        return Model(input_layer, [predictor_latent_mean, prediction_log_var])

    def make_latent_encoder(self):
        orig_batch_shape = (self.batch_size, self.original_dim)
        predictor_batch_shape = (self.batch_size, self.predictor_out_dim)
        # orig_batch_shape = (self.original_dim,)
        # predictor_batch_shape = (self.predictor_out_dim,)

        input_layer = Input(batch_shape = orig_batch_shape, 
                            name = 'input_layer')
        prediction_input_layer = Input(batch_shape = predictor_batch_shape, 
                            name = 'predictor_layer')

        input_w_pred = concatenate([input_layer, prediction_input_layer], axis = -1,
                                    name = 'input_w_pred_layer')
        
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
        
        latent_enc_input = [input_layer, prediction_input_layer]
        latent_enc_output = [vae_latent_mean, vae_latent_log_var]

        return Model(latent_enc_input, latent_enc_output)

    def make_latent_decoder(self, use_prev_input=False):

        input_batch_shape = (self.batch_size, self.original_dim)
        predictor_batch_shape = (self.batch_size, self.predictor_out_dim)
        vae_batch_shape = (self.batch_size, self.vae_latent_dim)
        # input_batch_shape = (self.original_dim,)
        # predictor_batch_shape = (self.predictor_out_dim,)
        # vae_batch_shape = (self.vae_latent_dim,)

        prediction_input_layer = Input(batch_shape=predictor_batch_shape, name='predictor_layer')
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

        pred_w_latent = concatenate([prediction_input_layer, prev_w_vae_latent], axis = -1)

        # build physical decoder
        vae_decoded_mean = self.model.get_layer('vae_decoded_mean')
        if self.vae_hidden_dim > 0:
            vae_dec_hid_layer = self.model.get_layer('vae_dec_hid_layer')
            vae_dec_hid_layer = vae_dec_hid_layer(pred_w_latent)
            vae_decoded_mean = vae_decoded_mean(vae_dec_hid_layer)
        else:
            vae_decoded_mean = vae_decoded_mean(pred_w_latent)

        if use_prev_input or self.use_prev_input:
            dec_input_stack = [prediction_input_layer, vae_latent_layer, prev_input_layer]
        else:
            dec_input_stack = [prediction_input_layer, vae_latent_layer]

        return Model(dec_input_stack, vae_decoded_mean)

    def sample_prediction(self, predictor_latent_mean, prediction_log_var, nsamps=1, nrm_samp=False, add_noise=True):
        
        if nsamps == 1:
            eps = np.random.randn(*((1, predictor_latent_mean.flatten().shape[0])))
        else:
            eps = np.random.randn(*((nsamps,) + predictor_latent_mean.shape))
        if eps.T.shape == predictor_latent_mean.shape:
            eps = eps.T
        if add_noise:
            predictor_norm = predictor_latent_mean + np.exp(prediction_log_var/2)*eps
        else:
            predictor_norm = self.predictor_latent_mean + 0*np.exp(self.prediction_log_var/2)*eps

        if nrm_samp: return predictor_norm

        if nsamps == 1:
            predictor_norm = np.hstack([predictor_norm, np.zeros((predictor_norm.shape[0], 1))])
            output = np.exp(predictor_norm)/np.sum(np.exp(predictor_norm), axis = -1)
            return output[:,None]
        else:
            predictor_norm = np.dstack([predictor_norm,np.zeros(predictor_norm.shape[:-1]+(1,))])
            output = np.exp(predictor_norm)/np.sum(np.exp(predictor_norm), axis = -1)
            return output[:,:,None]

    def sample_latent(self, Z_mean, Z_log_var, nsamps = 1):
        if nsamps == 1:
            eps = np.random.randn(*Z_mean.squeeze().shape)
        else:
            eps = np.random.randn(*((nsamps,) + Z_mean.squeeze().shape))
        return Z_mean + np.exp(Z_log_var/2) * eps

    def sample_vae(self, predictor_latent_mean):
        rando = np.random.rand(len(predictor_latent_mean.squeeze()))

        return np.float32(rando <= predictor_latent_mean)
