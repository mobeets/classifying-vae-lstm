# from https://github.com/philippesaade11/vaelstmclassifier/blob/GeneticAlgorithm/Genetic-Algorithm.py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from keras import backend as K
from keras.utils import to_categorical

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmclassifier.utils.model_utils import get_callbacks, init_adam_wn
from vaelstmclassifier.utils.model_utils import save_model_in_pieces
from vaelstmclassifier.utils.model_utils import AnnealLossWeight
from vaelstmclassifier.utils.data_utils import MNISTData
from vaelstmclassifier.utils.weightnorm import data_based_init
from vaelstmclassifier.vae_classifier.model import VAEClassifier
from vaelstmclassifier.vae_classifier.train import train_vae_classifier

class BlankClass(object):
    def __init__(self):
        pass

def generate_random_chromosomes(size, clargs, data_instance):
    generationID = 0
    nets = []
    for chromosomeID in range(size):
        chrom = Chromosome(clargs, data_instance, generationID, chromosomeID)
        chrom.train()
        nets.append(chrom)
    return nets

def select_parents(generation):
    total_fitness = sum(chrom.fitness for chrom in generation)
    #Generate two random numbers between 0 and total_fitness 
    #   not including total_fitness
    rand_parent1 = random.random()*total_fitness
    rand_parent2 = random.random()*total_fitness
    parent1 = None
    parent2 = None
    
    fitness_count = 0
    for chromosome in generation:
        fitness_count += chromosome.fitness
        if(parent1 == None and fitness_count >= rand_parent1):
            parent1 = chromosome
        if(parent2 == None and fitness_count >= rand_parent2):
            parent2 = chromosome
        if(parent1 != None and parent2 != None):
            break

    return parent1, parent2

def cross_over(parent1, parent2, prob):
    if(random.random() <= prob):
        params1 = {}
        params2 = {}
        for param in Chromosome.params:
            if(random.random() <= 0.5):
                params1[param] = parent1.params_dict[param]
                params2[param] = parent2.params_dict[param]
            else:
                params1[param] = parent2.params_dict[param]
                params2[param] = parent1.params_dict[param]
        
        clargs = parent1.clargs
        data_instance = parent1.data_instance
        generationID = parent1.generationID + 1
        chromosomeID = parent1.chromosomeID
        vae_kl_weight = parent1.vae_kl_weight
        predictor_weight = parent1.predictor_weight
        predictor_kl_weight = parent1.predictor_kl_weight

        child1 = Chromosome(clargs=clargs, data_instance=data_instance, 
            generationID=generationID, chromosomeID=chromosomeID,
            vae_kl_weight = vae_kl_weight, predictor_weight = predictor_weight,
            predictor_kl_weight = predictor_kl_weight, **params1)
        child2 = Chromosome(clargs=clargs, data_instance=data_instance, 
            generationID=generationID, chromosomeID=chromosomeID,
            vae_kl_weight = vae_kl_weight, predictor_weight = predictor_weight,
            predictor_kl_weight = predictor_kl_weight, **params2)

        return child1, child2
    
    return parent1, parent2

def mutate(child, prob):
    for param in Chromosome.params:
        if(random.random() <= prob):
            extra = int(child.params_dict[param]*0.1)+1
            child.params_dict[param] += random.randint(-extra, extra)
    return child

class Chromosome(VAEClassifier):
    
    #[number of hidden layers in VAE,
    #   size of the first hidden layer in VAE,
    #   size of the latent layer,
    #   number of hidden layers in the DNN regressor,
    #   size of the first hidden layer in the DNN regressor]
    params = ["size_vae_hidden", "size_vae_latent", 
                "size_dnn_latent", "size_dnn_hidden"]

    #If any of the parameters is set to -1, a random number if chosen
    def __init__(self, clargs, data_instance, generationID, chromosomeID, 
                size_vae_hidden = None, size_vae_latent = None, 
                size_dnn_hidden = None, size_dnn_latent = None,
                vae_kl_weight = 1.0, predictor_weight=1.0, 
                predictor_kl_weight = 1.0, verbose = False):
        
        self.verbose = verbose
        self.clargs = clargs
        self.data_instance = data_instance
        self.generationID = generationID
        self.chromosomeID = chromosomeID
        self.time_stamp = clargs.time_stamp

        self.vae_kl_weight = vae_kl_weight
        self.predictor_weight = predictor_weight
        self.predictor_kl_weight = predictor_kl_weight

        # num_vae_hidden = num_vae_hidden or random.randint(1, 10)
        size_vae_hidden = size_vae_hidden or random.randint(5, 10)
        size_vae_latent = size_vae_latent or random.randint(1, 10)    
        
        # num_dnn_hidden = num_dnn_hidden or random.randint(1, 10)
        size_dnn_latent = size_dnn_latent or random.randint(1, 10)
        size_dnn_hidden = size_dnn_hidden or random.randint(5, 10)

        self.params_dict = {#"num_vae_hidden": num_vae_hidden,
                           "size_vae_hidden": size_vae_hidden,
                           "size_vae_latent": size_vae_latent,
                           #"num_dnn_hidden": num_dnn_hidden,
                           "size_dnn_hidden": size_dnn_hidden,
                           "size_dnn_latent":size_dnn_latent}

        self.network_type = clargs.network_type
        self.original_dim = clargs.original_dim
        self.predictor_weight = clargs.predictor_weight
        
        self.optimizer = clargs.optimizer
        self.batch_size = clargs.batch_size
        self.use_prev_input = False
        self.class_dim = clargs.n_classes

        self.clf_hidden_dim = 2**size_dnn_hidden
        self.clf_latent_dim = 2**size_dnn_latent
        self.vae_hidden_dim = 2**size_vae_hidden
        self.vae_latent_dim = 2**size_vae_latent

        self.get_model()
        self.neural_net = self.model
        self.fitness = 0
        print('self.class_dim',self.class_dim)

    def train(self, verbose = False):
        """Training control operations to create VAEClassifier instance, 
            organize the input data, and train the network.
        
        Args:
            clargs (object): command line arguments from `argparse`
                Structure Contents: n_classes,
                    run_name, patience, kl_anneal, do_log, do_chkpt, num_epochs
                    w_kl_anneal, optimizer, batch_size
            
            data_instance (object): 
                Object instance for organizing data structures
                Structure Contents: train_classes, valid_classes, test_classes
                    labels_train, data_train, labels_valid, data_valid
        """
        verbose = verbose or self.verbose
        
        DI = self.data_instance
        clargs = self.clargs

        clf_train = to_categorical(DI.train_classes, self.clargs.n_classes)
        clf_validation = to_categorical(DI.valid_classes,self.clargs.n_classes)

        min_epoch = max(self.clargs.kl_anneal, self.clargs.w_kl_anneal)+1
        callbacks = get_callbacks(self.clargs, patience=self.clargs.patience, 
                    min_epoch = min_epoch, do_log = self.clargs.do_log, 
                    do_chckpt = self.clargs.do_chckpt)

        if clargs.kl_anneal > 0: 
            self.vae_kl_weight = K.variable(value=0.1)
        if clargs.w_kl_anneal > 0: 
            self.predictor_kl_weight = K.variable(value=0.0)
        
        clargs.optimizer, was_adam_wn = init_adam_wn(clargs.optimizer)
        clargs.optimizer = 'adam' if was_adam_wn else clargs.optimizer
        
        save_model_in_pieces(self.model, self.clargs)
        
        vae_train = DI.data_train
        vae_features_val = DI.data_valid

        data_based_init(self.model, DI.data_train[:clargs.batch_size])

        vae_labels_val = [DI.labels_valid, clf_validation, 
                            clf_validation,DI.labels_valid]

        validation_data = (vae_features_val, vae_labels_val)
        train_labels = [DI.labels_train, clf_train, clf_train, DI.labels_train]

        history = self.model.fit(vae_train, train_labels,
                                    shuffle = True,
                                    epochs = clargs.num_epochs,
                                    batch_size = clargs.batch_size,
                                    callbacks = callbacks,
                                    validation_data = validation_data)

        max_kl_anneal = max(clargs.kl_anneal, clargs.w_kl_anneal)
        best_ind = np.argmin([x if i >= max_kl_anneal + 1 else np.inf \
                    for i,x in enumerate(history.history['val_loss'])])
        
        best_loss = {k: history.history[k][best_ind] for k in history.history}
        
        self.fitness = 1.0 / best_loss
        # if(self.fitness < 0): self.fitness = 0

        if verbose: print('\n\n[INFO] The Best Loss: {}\n'.format(best_loss))
        
        joblib_save_loc = '{}/{}_{}_{}_trained_model_output_{}.joblib.save'
        joblib_save_loc = joblib_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)

        wghts_save_loc = '{}/{}_{}_{}_trained_model_weights_{}.save'
        wghts_save_loc = weights_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)
        
        model_save_loc = '{}/{}_{}_{}_trained_model_full_{}.save'
        model_save_loc = model_save_loc.format(self.model_dir, self.run_name,
                                         self.generationID, self.chromosomeID,
                                         self.time_stamp)
        
        vae_model.model.save_weights(wghts_save_loc, overwrite=True)
        vae_model.model.save(model_save_loc, overwrite=True)
        joblib.dump({'best_loss':best_loss,'history':history}, joblib_save_loc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str, default='ga_test_',
                help='tag for current run')
    parser.add_argument('--network_type', type=str, default="classification",
                help='select `classification` or `regression`')
    parser.add_argument('--batch_size', type=int, default=128,
                help='batch size')
    parser.add_argument('--optimizer', type=str, default='adam',
                help='optimizer name') 
    parser.add_argument('--num_epochs', type=int, default=200,
                help='number of epochs')
    parser.add_argument('--predictor_weight', type=float, default=1.0,
                help='relative weight on classifying key')
    parser.add_argument('--prediction_log_var_prior', type=float, default=0.0,
                help='w log var prior')
    parser.add_argument("--do_log", action="store_true", 
                help="save log files")
    parser.add_argument("--do_chckpt", action="store_true",
                help="save model checkpoints")
    parser.add_argument('--patience', type=int, default=10,
                help='# of epochs, for early stopping')
    parser.add_argument("--kl_anneal", type=int, default=0, 
                help="number of epochs before kl loss term is 1.0")
    parser.add_argument("--w_kl_anneal", type=int, default=0, 
                help="number of epochs before w's kl loss term is 1.0")
    parser.add_argument('--log_dir', type=str, default='data/logs',
                help='basedir for saving log files')
    parser.add_argument('--model_dir', type=str, default='data/models',
                help='basedir for saving model weights')    
    parser.add_argument('--train_file', type=str, default='MNIST',
                help='file of training data (.pickle)')
    parser.add_argument('--verbose', action='store_true',
                help='print more [INFO] and [DEBUG] statements')

    clargs = parser.parse_args()
    
    verbose = clargs.verbose

    clargs.data_type = 'MNIST'
    data_instance = MNISTData(batch_size = clargs.batch_size)
    
    n_train, n_features = data_instance.data_train.shape
    n_test, n_features = data_instance.data_valid.shape

    clargs.original_dim = n_features
    
    clargs.time_stamp = int(time())
    clargs.run_name = '{}_{}_{}'.format(clargs.run_name, 
                                clargs.data_type, clargs.time_stamp)

    if verbose: print('\n\n[INFO] Run Base Name: {}\n'.format(clargs.run_name))
    
    clargs.n_classes = len(np.unique(data_instance.train_classes))

    cross_prob = 0.7
    mutate_prob = 0.01
    net_size = 10  #Preferably divisible by 2

    generation = generate_random_chromosomes(net_size,
            clargs = clargs, data_instance = data_instance)
    gen_num = 0

    best_fitness = []
    fig = plt.gcf()
    fig.show()

    iterations = 500
    while gen_num < iterations:

        #Create new generation
        new_generation = []
        gen_num += 1
        for _ in range(int(net_size/2)):
            parent1, parent2 = select_parents(generation)
            child1, child2 = cross_over(parent1, parent2, cross_prob)
            
            mutate(child1, mutate_prob)
            mutate(child2, mutate_prob)
            
            child1.train()
            child2.train()
            
            new_generation.append(child1)
            new_generation.append(child2)
        generation = new_generation

        best_fitness.append(max(chrom.fitness for chrom in generation))
        plt.plot(best_fitness, color="c")
        plt.xlim([0, iterations])
        fig.canvas.draw()