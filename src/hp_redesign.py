import torch
import pyro
import numpy as np

from pyro.distributions import Dirichlet, Gamma, Multinomial
from typing import Dict

class DirichletProcess:
    def __init__(self, alpha: float, base_distribution: Dict = None):
        '''
        Initialize a Dirichlet Process with concentration parameter alpha

        '''
        self.alpha = alpha
        if (base_distribution is None):
            self.base_distribution = Dirichlet(torch.ones(1))
            self.base_values = list(range(1))
        else:
            self.base_distribution = Multinomial(1, base_distribution["weights"])
            self.base_values = base_distribution["values"]
        self.entries = {}

    def sample(self):
        '''
        Sample from the Dirichlet Process
        '''
        # Sample from the base distribution with probability alpha / (alpha + N)
        total_counts = sum(self.entries.values())
        probs = list(self.entries.values()) / total_counts
        entries = list(self.entries.keys())
        p_existing = self.alpha / (self.alpha + total_counts)
        if (np.random.rand() < p_existing):
            # Select existing customer
            idx = Multinomial(1, probs).sample(1).item()
            return entries[idx]
        else:
            # Sample from the base distribution
            new_entry = self.base_distribution.sample()
            if new_entry not in self.entries:
                self.entries[new_entry] = 1
            else:   
                self.entries[new_entry] += 1
            return new_entry
    
    def add_entry(self, entry):
        '''
        Add an entry to the Dirichlet Process
        '''
        if entry not in self.entries:
            self.entries[entry] = 1
        else:
            self.entries[entry] += 1


class HierarchicalDirichletProcess(DirichletProcess):
    def __init__(self, alpha, base_distribution, layers):
        super().__init__(alpha, base_distribution)
        self.layers = layers

    def sample(self):
        # Sample from the DP to determine which base distribution to use
        base_distribution_idx = super().sample()

        # Sample from the selected base distribution
        base_distribution = self.entries[base_distribution_idx]

        # Sample from the base distribution to determine concentration parameter
        concentration = base_distribution.rvs(self.base_concentration)

        # Sample from the base distribution with concentration parameter
        sample = base_distribution.rvs(concentration)
        self.customers_concentration.append(concentration)

        return sample
