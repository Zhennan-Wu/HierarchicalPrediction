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
    def __init__(self, gamma_param1, gamma_param2, base, base_size, batch_size, sample_size = 100):
        self.GAMMA = dist.Gamma(torch.tensor(gamma_param1), torch.tensor(gamma_param2))
        self.Gs3 = {}
        self.Gs2 = {}
        self.Gs1 = {}
        self.Gb = {}
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.tau = torch.ones(base_size)
        self.generate_concenctration()
    
    def generate_concenctration(self):
        '''
        Generate concentration parameters for the Dirichlet distribution
        '''
        # Concentration parameters for the Dirichlet distribution
        self.alpha1 = self.GAMMA.sample()
        self.alpha2 = self.GAMMA.sample()
        self.alpha3 = self.GAMMA.sample()

        
        self.beta = self.GAMMA.sample()
        self.gamma = self.GAMMA.sample()
        self.eta = None
        self.generate_prior()

    def generate_prior(self):
        self.base = dist.Dirichlet(self.beta*self.tau)

    def nested_CRP(self, sample_size = None):
        '''
        Nested Chinese restaurant process to generate two level hierarchical DP
        '''
        # Sample distribution parameter from prior for super category class
        eta = self.GAMMA.sample()

        # A dictionary with key to be the different super categories and value to be the count of each super category
        super_category_collection = {}

        # A dictionary with key to be the different super categories and value to be the topic prior of each super category
        topic_prior_collection = {}

        # A dictionary with key to be the different super categories and value to be a dictionary with key to be the different base categories and value to be the count of each base category
        base_category_collection = {} 

        # A dictionary with key to be the different super categories and value to be a dictionary with key to be the different base categories and value to be the topic distribution of each base category
        topic_distribution_collection = {}

        # A dictionary with key to be the different super categories and value to be a dictionary with key to be the different base categories and value to be the prior of each base category
        base_prior_collection = {}

        # A dictionary with key to be the different super categories and value to be a dictionary with key to be the different base categories and value to be a dictionary with key to be the different topics and value to be the word distribution of each topic
        word_distribution_collection = {}

        # Keep track of the maximum number of base categories
        base_category_count = 0

        if (sample_size == None):
            sample_size = np.square(self.sample_size)

        for _ in range(sample_size):
            # Select super category label
            values = list(super_category_collection.keys())
            count = list(super_category_collection.values()) + [eta]
            probs = torch.tensor(count) / sum(count)
            index = dist.Categorical(probs).sample()
            if index < len(values):
                super_cat = values[index]
            else:
                # Draw new super category from global distribution
                super_cat = dist.Dirichlet(self.eta*self.tau).sample()
            if (super_cat in super_category_collection):
                super_category_collection[super_cat] += 1
            else:
                super_category_collection[super_cat] = 1
                base_category_collection[super_cat] = {}
                # Sample distribution parameter from prior for one base category
                base_prior_collection[super_cat] = self.GAMMA.sample()

            # Select base category label under the selected super category
            base_category_values = list(base_category_collection[super_cat].keys())
            base_count = list(base_category_collection[super_cat].values()) + [base_prior_collection[super_cat]]
            base_probs = torch.tensor(base_count) / sum(base_count)
            base_index = dist.Categorical(base_probs).sample()
            if base_index < len(base_category_values):
                base_cat = base_category_values[base_index]
            else:   
                # Draw new base category base category prior of the selected super category
                base_cat = dist.Dirichlet(base_prior_collection[super_cat]*self.tau).sample()
            if (base_cat in base_category_collection[super_cat]):
                base_category_collection[super_cat][base_cat] += 1
            else:
                base_category_collection[super_cat][base_cat] = 1
            
            # Update the maximum number of base categories
            base_category_count = max(base_category_count, len(base_category_collection[super_cat].keys()))
        
        return base_category_collection, base_category_count
            
    def dirichlet_process(self, alpha, base = None, sample_size = None):
        '''
        Chinese restaurant process generating a Dirichlet process
        '''
        value_collection = {}
        if (sample_size == None):
            sample_size = self.sample_size
        for _ in range(sample_size):
            values = list(value_collection.keys())
            count = list(value_collection.values()) + [alpha]
            probs = torch.tensor(count) / sum(count)
            index = dist.Categorical(probs).sample()
            if index < len(values):
                sample = values[index]
            else:
                if (base == None):
                    sample = self.base.sample()
                else:
                    sample = base.sample() 
            if (sample in value_collection):
                value_collection[sample] += 1
            else:
                value_collection[sample] = 1
        return value_collection
    
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
