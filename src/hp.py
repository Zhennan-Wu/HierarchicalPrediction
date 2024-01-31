import argparse
import functools
import logging

import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


class HP:
    '''
    Hierarchical Prior
    '''
    def __init__(self, alpha1, alpha2, alpha3, beta, gamma, base, batch_size):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.beta = beta
        self.gamma = gamma
        self.eta = None
        self.base = base
        self.Gs3 = {}
        self.Gs2 = {}
        self.Gs1 = {}
        self.Gb = {}
        self.batch_size = batch_size
    
    def nested_CRP(self):
        '''
        Nested Chinese restaurant process to generate two level hierarchical DP
        '''
        super_category_history = {}
        eta = "sample from gamma prior"
        super_categories_index = 0
        base_category_collection = {}
        base_prior_collection = {}
        base_index_collection = {}
        for _ in range(self.batch_size):
            # Super category
            values = list(super_category_history.keys())
            count = list(super_category_history.values()) + [eta]
            probs = torch.tensor(count) / sum(count)
            index = torch.multinomial(probs, 1, replacement=True)
            if index < len(values):
                sample = values[index]
            else:
                sample = super_categories_index
                super_categories_index += 1
            if (sample in super_category_history):
                super_category_history[sample] += 1
            else:
                super_category_history[sample] = 1
                super_categories_index += 1
                base_category_collection[sample] = {}
                base_prior_collection[sample] = "sample from gamma prior"
                base_index_collection[sample] = 0

            # Base category
            base_category_values = list(base_category_collection[sample].keys())
            base_count = list(base_category_collection[sample].values()) + [base_prior_collection[sample]]
            base_probs = torch.tensor(base_count) / sum(base_count)
            base_index = torch.multinomial(base_probs, 1, replacement=True)
            if base_index < len(base_category_values):
                base_sample = base_category_values[base_index]
            else:   
                base_sample = base_index_collection[sample]
                base_index_collection[sample] += 1    
            if (base_sample in base_category_collection[sample]):
                base_category_collection[sample][base_sample] += 1
            else:
                base_category_collection[sample][base_sample] = 1
                base_index_collection[sample] += 1
        
        return super_category_history, base_category_collection
            
    def dirichlet_process(self, n, alpha, base = None):
        '''
        Chinese restaurant process generating a Dirichlet process
        '''
        value_history = {}
        for _ in range(n):
            values = list(value_history.keys())
            count = list(value_history.values()) + [alpha]
            probs = torch.tensor(count) / sum(count)
            index = torch.multinomial(probs, 1, replacement=True)
            if index < len(values):
                sample = values[index]
            else:
                if (base == None):
                    sample = self.base.sample()
                else:
                    sample = base.sample() 
            if (sample in value_history):
                value_history[sample] += 1
            else:
                value_history[sample] = 1
        return value_history
    
    def dirichlet_process_posterior(self, n, observation, alpha, base):
        '''
        From the posterior distribution of the Dirichlet process to update parameters
        '''
        observation_count = sum(list(observation.values()))
        new_alpha = alpha + observation_count
        value_history = {}
        for _ in range(n):
            values = list(value_history.keys())
            count = list(value_history.values()) + [new_alpha]
            probs = torch.tensor(count) / sum(count)
            index = torch.multinomial(probs, 1, replacement=True)
            if index < len(values):
                sample = values[index]
            else:
                base_prob = torch.tensor([alpha/new_alpha, observation_count/new_alpha])
                base_index = torch.multinomial(base_prob, 1, replacement=True)
                if (base_index == 0):
                    sample = self.base.sample()
                else:
                    dist.Categorical(torch.tensor(list(observation.values()))/sum(list(observation.values()))).sample()
            if (sample in value_history):
                value_history[sample] += 1
            else:
                value_history[sample] = 1
        return value_history
    
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

    def hierarchical_dp_sample_generation(self, Gg3):
        '''
        Generate DBM top level latent variables
        '''

        Gs2 = self.dirichlet_process(5, self.alpha3, dist.Categorical(torch.tensor(list(Gg3.values()))/sum(list(Gg3.values()))))

        Gs1 = self.dirichlet_process(5, self.alpha2, dist.Categorical(torch.tensor(list(Gs2.values()))/sum(list(Gs2.values()))))

        Gb = self.dirichlet_process(5, self.alpha1, dist.Categorical(torch.tensor(list(Gs1.values()))/sum(list(Gs1.values()))))

        phi_n = dist.Categorical(torch.tensor(list(Gb.values()))/sum(list(Gb.values())))
        h3  = phi_n.sample()

        return h3
