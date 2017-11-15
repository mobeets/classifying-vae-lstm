import json
import os.path
import numpy as np
from keras import losses
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from utils.weightnorm import AdamWithWeightnorm, data_based_init

bincrossentropy = lambda x, xhat: (x*np.log(np.maximum(1e-15, xhat)) + (1-x)*np.log(np.maximum(1e-15, 1-xhat)))

def logmeanexp(vs, axis=0):
    m = np.amax(vs, axis=axis)
    return m + np.log(np.mean(np.exp(vs - m[None,:]), axis=axis))

def logsumexp(vs, axis=0):
    m = np.amax(vs, axis=axis)
    return m + np.log(np.sum(np.exp(vs - m[None,:]), axis=axis))

class AnnealLossWeight(Callback):
    """
    increase the weight of a loss term by adjusting its value as a function of the epoch number
    """
    def __init__(self, beta, name="beta", n_epochs=10, final_value=1.0, slope=0):
        super(AnnealLossWeight, self).__init__()
        self.beta = beta
        self.name = name
        self.slope = slope
        self.n_epochs = n_epochs
        self.start_value = K.eval(beta)
        self.final_value = final_value
        self.all_done = False

    def next_weight(self, x):
        if self.slope > 0:
            # sigmoid between 0.0 and 1.0 given x between 0.0 and 1.0
            return 1 / (1 + np.exp(-self.slope*(x-0.5)))
        else:
            # linear
            return 1.0*x

    def on_epoch_begin(self, epoch, logs={}):
        if self.all_done:
            return
        if epoch >= self.n_epochs:
            next_val = self.final_value
            self.all_done = True
        else:
            next_val = self.start_value + self.next_weight(1.0*epoch/self.n_epochs)*(self.final_value - self.start_value)
        K.set_value(self.beta, next_val)
        print "+++++ {}: {}".format(self.name, K.eval(self.beta))

def init_adam_wn(optimizer):
    if optimizer == 'adam-wn':
        adam_wn = AdamWithWeightnorm(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        return adam_wn, True
    else:
        return optimizer, False

class EarlyStoppingAfterEpoch(Callback):
    def __init__(self, monitor='val_loss', min_epoch=0, min_delta=0, patience=0, verbose=0, mode='auto'):
        super(EarlyStoppingAfterEpoch, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_epoch = min_epoch
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        assert mode in ['auto', 'min', 'max']

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epoch:
            return
        current = logs.get(self.monitor)
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

class ModelCheckpointAfterEpoch(Callback):
    def __init__(self, filepath, monitor, min_epoch=0, save_weights_only=True, save_best_only=True, mode='auto', verbose=False):
        super(ModelCheckpointAfterEpoch, self).__init__()
        assert save_best_only and not verbose
        assert mode in ['auto', 'min', 'max']
        self.filepath = filepath
        self.monitor = monitor
        self.min_epoch = min_epoch
        self.save_weights_only = save_weights_only
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.min_epoch:
            return
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)

def get_callbacks(args, patience=5, min_epoch=0, do_log=False):
    # prepare to save model checkpoints
    chkpt_filename = os.path.join(args.model_dir, args.run_name + '.h5')
    checkpt = ModelCheckpointAfterEpoch(chkpt_filename, min_epoch=min_epoch,
        monitor='val_loss', save_weights_only=True, save_best_only=True)
    # checkpt = ModelCheckpoint(chkpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True)
    callbacks = [checkpt]
    if do_log:
        logging = TensorBoard(log_dir=os.path.join(args.log_dir, args.run_name))
        callbacks.append(logging)
    if patience > 0:
        early_stop = EarlyStoppingAfterEpoch(monitor='val_loss',
            min_epoch=min_epoch, patience=patience, verbose=0)
        callbacks.append(early_stop)
        # early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=0)
        callbacks.append(early_stop)
    return callbacks

def save_model_in_pieces(model, args):
	# save model structure
    outfile = os.path.join(args.model_dir, args.run_name + '.yaml')
    with open(outfile, 'w') as f:
        f.write(model.to_yaml())
    # save model args
    outfile = os.path.join(args.model_dir, args.run_name + '.json')
    json.dump(vars(args), open(outfile, 'w'))

def LL_frame(y, yhat):
    return 88*losses.binary_crossentropy(y, yhat)
