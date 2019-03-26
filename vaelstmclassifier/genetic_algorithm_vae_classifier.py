# from https://github.com/philippesaade11/vaelstmclassifier/blob/GeneticAlgorithm/Genetic-Algorithm.py
import random
import matplotlib.pyplot as plt

from vaelstmclassifier.vae_classifier.model import VAEClassifier

import argparse
import os

from glob import glob
from numpy import array, arange, vstack, reshape, loadtxt, zeros
from sklearn.externals import joblib
from time import time
from tqdm import tqdm

from vaelstmclassifier.vae_classifier.train import train_vae_classifier
# vae_classifier_train = train.train_vae_classifier # rename

class BlankClass(object):
    def __init__(self):
        pass

def generate_random_chromosomes(size):
    nets = []
    for _ in range(size):
        chrom = Chromosome()
        chrom.train()
        nets.append(chrom)
    return nets

def select_parents(generation):
    total_fitness = sum(chrom.fitness for chrom in generation)
    #Generate 2 random numbers between 0 and total_fitness not including total_fitness
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
            else :
                params1[param] = parent2.params_dict[param]
                params2[param] = parent1.params_dict[param]
                
        child1 = Chromosome(**params1)
        child2 = Chromosome(**params2)
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
    def __init__(self, original_dim, n_classes,
                size_vae_hidden = None, size_vae_latent = None, 
                size_dnn_hidden = None, size_dnn_latent = None,
                batch_size = 128, vae_kl_weight = 1.0, clf_weight=1.0, 
                clf_kl_weight = 1.0, optimizer = 'adam', 
                use_prev_input = False, network_type = 'classification'):

        # num_h_VAE=-1, num_h_DNN=-1, 
        # if(num_h_VAE == -1):
        #     num_h_VAE = random.randint(1, 10)
        if size_vae_hidden is None:
            size_vae_hidden = random.randint(5, 10)
        if size_vae_latent is None:
            size_vae_latent = random.randint(1, 10)
        # if(num_h_DNN == -1):
        #     num_h_DNN = random.randint(1, 10)
        if size_dnn_latent is None:
            size_dnn_latent = random.randint(1, 10)
        if size_dnn_hidden is None:
            size_dnn_hidden = random.randint(5, 10)

        self.params_dict = {#"num_h_VAE": num_h_VAE,
                           "size_vae_hidden": size_vae_hidden,
                           "size_vae_latent": size_vae_latent,
                           "size_dnn_latent":size_dnn_latent,
                           #"num_h_DNN": num_h_DNN,
                           "size_dnn_hidden": size_dnn_hidden}

        self.network_type = network_type
        self.original_dim = original_dim
        
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.use_prev_input = use_prev_input
        
        self.class_dim = n_classes
        self.clf_hidden_dim = 2**size_dnn_hidden
        self.clf_latent_dim = 2**size_dnn_latent
        self.vae_hidden_dim = 2**size_vae_hidden
        self.vae_latent_dim = 2**size_vae_latent

        self.get_model()
        self.neural_net = self.model
        self.fitness = 0

    def train(self):
        self.fitness = -self.params_dict["size_1st_VAE"]*self.params_dict["num_h_DNN"]*0.5-self.params_dict["size_1st_DNN"]*self.params_dict["num_h_VAE"]+self.params_dict["num_h_VAE"]*3+self.params_dict["size_1st_VAE"]*1.2+self.params_dict["size_vae_latent"]+self.params_dict["num_h_DNN"]*5+self.params_dict["size_1st_DNN"]*2.5+random.uniform(0, 10)
        if(self.fitness < 0):
            self.fitness = 0

cross_prob = 0.7
mutate_prob = 0.01
net_size = 10  #Preferably divisible by 2

generation = generate_random_chromosomes(net_size)
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
        
            
