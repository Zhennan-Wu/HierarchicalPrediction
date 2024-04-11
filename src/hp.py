import argparse
import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import numpy as np
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


class HP:
    '''
    Hierarchical Prior
    '''
    def __init__(self, gamma_param1, gamma_param2, num_topic, num_word, batch, batch_size, sample_size = 100):
        '''
        '''
        self.GAMMA = dist.Gamma(torch.tensor(gamma_param1), torch.tensor(gamma_param2))
        self.batch = batch
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_topic = num_topic
        self.num_word = num_word

        self.init_concenctration()
        self.pi_g, self.tau = self.generate_prior(num_topic, num_word)
        self.init_super_category()
    
    def generate_prior(self, topic_dim, word_dim):
        '''
        Generate prior distribution for the model
        '''
        pi_g = dist.Dirichlet(torch.ones(topic_dim)).sample()
        tau = dist.Dirichlet(torch.ones(word_dim)).sample()
        return pi_g, tau

    def init_concenctration(self):
        '''
        Generate concentration parameters for the Dirichlet distribution
        '''
        # Concentration parameters for the Dirichlet distribution
        self.alpha1 = self.GAMMA.sample()
        self.alpha2 = self.GAMMA.sample()
        self.alpha3 = self.GAMMA.sample()
        
        self.beta = self.GAMMA.sample()
        self.gamma = self.GAMMA.sample()

    def init_super_category(self):
        '''
        Initialize super category
        '''
        # Sample distribution parameter from prior for super category class
        self.eta = self.GAMMA.sample()

        # A dictionary with key to be the different super categories and value to be the count of each super category
        self.super_category_collection = {}

        # A dictionary with key to be the different super categories and value to be a dictionary with key to be the different base categories and value to be the count of each base category
        self.base_category_collection = {} 

        # Keep track of the number of super categories
        self.super_category_index = 0

        # Keep track of the number of base categories in each super category
        self.base_category_indices = {}

    def generate_prior(self):
        self.base = dist.Dirichlet(self.beta*self.tau)

    def nested_CRP(self):
        '''
        Nested Chinese restaurant process to generate two level hierarchical DP
        '''
        count = list(self.super_category_collection.values()) + [self.eta]
        probs = torch.tensor(count) / sum(count)
        super_cat = dist.Categorical(probs).sample() + 1
        # Generate new category
        if (super_cat == len(count)):
            self.super_category_index += 1
            self.super_category_collection[self.super_category_index] = 1
            self.base_category_indices[self.super_category_index] = 1
            self.base_category_collection[self.super_category_index] = {}
            self.base_category_collection[self.super_category_index][self.base_category_indices[self.super_category_index]] = 1
        # Update existing category
        else:
            self.super_category_collection[super_cat] += 1
            # Select base category label under the selected super category
            base_count = list(self.base_category_collection[super_cat].values()) + [self.eta]
            base_probs = torch.tensor(base_count) / sum(base_count)
            base_cat = dist.Categorical(base_probs).sample() + 1
            # Create new base category
            if (base_cat == len(base_count)):
                self.base_category_indices[super_cat] += 1
                self.base_category_collection[super_cat][base_cat] = 1
            # Update existing base category
            else:   
                self.base_category_collection[super_cat][base_cat] += 1
        return [super_cat, base_cat]
            
    def create_z_label(self):
        '''
        Generate latent variable z for each sample
        '''
        z = []
        for _ in range(self.batch_size):
            z.append(self.nested_CRP())
        return torch.tensor(z)
    
    def mixture_distribution(self, continous_part, discrete_part, cont_weight):
        '''
        Mixture distribution of continuous and discrete distribution
        '''
        cont_sample = continous_part.sample()
        disc_sample = discrete_part.sample()
        return cont_sample*cont_weight + disc_sample*(1-cont_weight)
    
    def dirichlet_process(self, alpha, base = None, sample_size = None):
        '''
        Chinese restaurant process generating a Dirichlet process
        '''
        value_collection = []
        value_indices = {}
        if (sample_size == None):
            sample_size = self.sample_size
        for _ in range(sample_size):
            values = list(value_indices.keys())
            count = list(value_indices.values()) + [alpha]
            probs = torch.tensor(count) / sum(count)
            index = dist.Categorical(probs).sample()
            if (index < len(values)):
                value_indices[index] += 1
            else:
                if (base == None):
                    sample = self.base.sample()
                else:
                    weights = base[0]
                    values = base[1]
                    sample = values[dist.Categorical(weights).sample()]
                if (sample in value_collection):
                    value_indices[value_collection.index(sample)] += 1
                else:
                    value_collection.append(sample)
                    value_indices[index] = 1
        weights = torch.tensor(list(value_indices.values()))/sum(list(value_indices.values()))
        return weights, value_collection
    
    def dirichlet_process_posterior(self, observation, alpha, base = None, sample_size = None):
        '''
        From the posterior distribution of the Dirichlet process to update parameters
        '''
        observation_count = sum(list(observation.values()))
        new_alpha = alpha + observation_count
        value_collection = {}
        if (sample_size == None):
            sample_size = self.sample_size
        for _ in range(sample_size):
            values = list(value_collection.keys())
            count = list(value_collection.values()) + [new_alpha]
            probs = torch.tensor(count) / sum(count)
            index = dist.Categorical(probs).sample()
            if index < len(values):
                sample = values[index]
            else:
                base_prob = torch.tensor([alpha/new_alpha, observation_count/new_alpha])
                base_index = dist.Categorical(base_prob).sample()
                if (base_index == 0):
                    sample = self.base.sample()
                else:
                    dist.Categorical(torch.tensor(list(observation.values()))/sum(list(observation.values()))).sample()
            if (sample in value_collection):
                value_collection[sample] += 1
            else:
                value_collection[sample] = 1
        return value_collection
    
    def generate_hdp(self):
        '''
        Generate hierarchical DP distribution reference table, need to be called after generate z labels
        '''
        super_category_dists = []
        base_category_dists = []
        for super_cat in range(self.super_category_index):
            base_category_prior = dist.Dirichlet(self.pi_g*self.alpha3).sample()
            super_category_dists.append(base_category_prior)
            topic_dists = []
            for _ in range(self.base_category_indices[super_cat]):
                topic_dists.append(dist.Dirichlet(base_category_prior*self.alpha2).sample())
            base_category_dists.append(topic_dists)
        
        word_dists = []
        for _ in range(self.num_topic):
            word_dists.append(dist.Dirichlet(self.tau*self.beta).sample())
        return super_category_dists, base_category_dists, word_dists

    def infer(self):
        '''
        '''
        # top down generate hidden variable h3
        z_labels = self.create_z_label()
        super_category_dists, base_category_dists, word_dists = self.generate_hdp()

        # vocab = [] # list h3 samples of all documents
        super_cat_lst = [] # list of super category labels of all documentss
        base_cat_lst = [] # list of base category labels of all documents
        topic_lst = [] # list of topic labels of h3 samples of all documents

        for i in range(self.batch_size):
            super_cat = z_labels[i][0].item()
            base_cat = z_labels[i][1].item()
            topic_dist = dist.Dirichlet(base_category_dists[super_cat][base_cat]*self.alpha1).sample()
            # words = []
            topics = []
            for j in range(self.num_word):
                topic = dist.Categorical(topic_dist).sample()
                topics.append(topic)
                # words.append(dist.Categorical(word_dists[topic]).sample())
            # vocab.append(words)
            super_cat_lst.append(super_cat)
            base_cat_lst.append(base_cat)
            topic_lst.append(topics)
        
        # compute topic posterior based on h3 sample values
        topic_log_posterior = torch.zeros(self.num_topic)
        for i in range(self.batch_size):
            for j in range(self.num_word):
                for t in range(self.num_topic):
                    topic_log_posterior[t] += dist.Categorical(word_dists[t]).log_prob(self.batch[i][j]) + dist.Categorical(base_category_dists[super_cat_lst[i]][base_cat_lst[i]]).log_prob(t)
        topic_posterior = torch.exp(topic_log_posterior)/torch.sum(torch.exp(topic_log_posterior))

        # compute category posterior based on topic posterior 
                
        
        # need to keep track of topic count
    
        
    def hierarchical_dp_weight_update(self, h3):
        '''
        Update the weight of the hierarchical DP based on the posterior distribution
        '''
        Gb = self.dirichlet_process_posterior(5, h3, self.alpha1, dist.Categorical(torch.tensor(list(self.Gb.values()))/sum(list(self.Gb.values()))))

        Gs1 = self.dirichlet_process_posterior(5, Gb, self.alpha2, dist.Categorical(torch.tensor(list(self.Gs1.values()))/sum(list(self.Gs1.values()))))

        Gs2 = self.dirichlet_process_posterior(5, Gs1, self.alpha3, dist.Categorical(torch.tensor(list(self.Gs2.values()))/sum(list(self.Gs2.values()))))

        Gg3 = self.dirichlet_process_posterior(5, Gs2, self.gamma, dist.Dirichlet(torch.tensor(self.beta)))

        # Update parameters
        self.Gg3 = Gg3
        self.Gs2 = Gs2
        self.Gs1 = Gs1
        self.Gb = Gb
        return Gg3

    def hierarchical_dp_sample_generation(self):
        '''
        Generate DBM top level latent variables
        '''
        base_category_collection, base_category_count = self.nested_CRP()

        topic_collection = []
        for _ in range(base_category_count):
            topic_collection.append(self.dirichlet_process(self.eta, dist.Categorical(torch.tensor(list(self.Gg3.values()))/sum(list(self.Gg3.values())))))

        Gg3 = self.dirichlet_process(self.gamma, self.base)

        Gs2 = self.dirichlet_process(self.alpha3, dist.Categorical(torch.tensor(list(Gg3.values()))/sum(list(Gg3.values()))))
        
        Gs1 = self.dirichlet_process(self.alpha2, dist.Categorical(torch.tensor(list(base_category_collection[Gs2].values())/sum(list(base_category_collection[Gs2].values())))))

        Gb = topic_collection[Gs1.keys()]

        phi_n = dist.Categorical(torch.tensor(list(Gb.values()))/sum(list(Gb.values())))
        h3  = phi_n.sample()

        return h3
    
    def topic_sampling(self, super_cat, base_cat):
        '''
        - Update the topic distribution for each base category under the super category
        - Sample a topic for each word in the document
        '''
        pass

    def init_hierarchy(self, batch):
        '''
        Initialize the hierarchy
        '''

        # assign hierarchy to each sample
        super_cats = []
        base_cats = []
        topics = []
        for data in batch:
            super_cat, base_cat = self.nested_CRP()
            super_cats.append(super_cat)
            base_cats.append(base_cat)
            topic = self.topic_sampling(super_cat, base_cat)
            topics.append(topic)

        # update the hierarchy
        

        