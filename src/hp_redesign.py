import torch
import pyro
import itertools

from pyro.distributions import Dirichlet, Gamma, Categorical
from torch.multiprocessing import Pool
from typing import Dict, Union


class Categorical_Distribution:
    def __init__(self, weights: list, values: list):
        '''
        Initialize a Categorical Distribution with weights

        Parameters:
        - weights (list): the weights of the values
        - values (list): the values of the Categorical Distribution
        '''
        self.weights = torch.tensor(weights)
        self.values = torch.tensor(values)
    
    def sample(self):
        '''
        Sample from the Categorical Distribution
        '''
        idx = Categorical(self.weights).sample()
        return self.values[idx]
    

class DirichletProcess:
    def __init__(self, alpha: float, base_distribution: Dict = None):
        '''
        Initialize a Dirichlet Process with concentration parameter alpha

        Parameters:
        - alpha (float): the concentration parameter of the Dirichlet Process
        - base_distribution (Dict): the base distribution of the Dirichlet Process
        '''
        self.alpha = alpha
        self.values = []
        self.weights = []
        if (base_distribution is None):
            # The dimension of the base distribution should be set according to the dimension of the top hidden layer in the Deep Boltzmann Machine
            beta = Gamma(1, 1).sample()
            self.base_distribution = Dirichlet(beta*torch.ones(10))
        else:
            self.base_distribution = Categorical_Distribution(base_distribution["weights"], base_distribution["values"])

    def sample(self, num_samples: int):
        '''
        Sample from the Dirichlet Process

        Parameters:
        - num_samples (int): the number of samples to draw
        '''
        # Sample from the base distribution with probability alpha / (alpha + N)
        for i in range(num_samples):
            total_counts = sum(self.weights)
            probs = torch.tensor(self.weights) / total_counts
            p_existing = self.alpha / (self.alpha + total_counts)
            if (torch.rand(1) > p_existing):
                # Select existing sample
                idx = Categorical(probs).sample()
                self.weights[idx] += 1
                new_entry = self.values[idx]
            else:
                # Sample from the base distribution
                new_entry = self.base_distribution.sample()
                self.values.append(new_entry)
                self.weights.append(1)
    
    def get_values(self):
        '''
        Get the entries in the Dirichlet Process
        '''
        return self.values
    
    def get_weights(self):
        '''
        Get the weights of the entries in the Dirichlet Process
        '''
        return self.weights

    def get_distribution(self):
        '''
        Get the distribution of the Dirichlet Process
        '''
        return {"values": self.values, "weights": self.weights}

class HierarchicalDirichletProcess:
    def __init__(self, layers: int):
        '''
        Initialize a Hierarchical Dirichlet Process with layers

        Parameters:
        - layers (int): the number of layers in the Hierarchical Dirichlet Process
        '''
        self.layers = layers
        self.fixed
        self.categories_per_layer = torch.zeros(layers, dtype=torch.int32)

    def summarize_CRP(self, labels: Union[torch.Tensor, list]):
        '''
        Summarize the Chinese Restaurant Process to get the unique values and their counts

        Parameters:
        - labels (torch.Tensor or list): the labels of the samples

        Returns:
        - unique_values (torch.Tensor): the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        - counts (torch.Tensor): the counts of the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        '''
        unique_values_list = []
        counts_list = []
        if (isinstance(labels, list)):
            for label in labels:
                unique_values, counts = torch.unique(label, return_counts=True)
                unique_values_list.append(unique_values)
                counts_list.append(counts)
            unique_values = torch.cat(unique_values_list, dim=0)
            counts = torch.cat(counts_list, dim=0)
        else:
            unique_values, counts = torch.unique(labels, return_counts=True)
        return unique_values, counts
    
    def generate_CRP(self, sample_size: int, eta: float):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta
        
        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - eta (float): the concentration parameter of the Chinese Restaurant Process

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        labels = torch.tensor([0])
        for _ in range(1, sample_size):
            unique_values, counts = torch.unique(labels, return_counts=True)
            if (torch.rand(1) < eta/(eta + torch.sum(counts))):
                # Add a new label
                new_label = torch.max(unique_values) + 1
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
            else:
                # Select an existing label
                new_label = Categorical(counts).sample().item()
                labels = torch.cat((labels, torch.tensor([new_label])), dim=0)
        return labels

    def generate_fixed_categories(self, sample_size: int, eta: float, num_categories: int):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta
        
        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - num_categories (int): the number of categories to generate

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        labels = torch.tensor([0])
        for _ in range(1, sample_size):
            unique_values, counts = torch.unique(labels, return_counts=True)
            new_label = torch.max(unique_values) + 1
            if (torch.rand(1) < eta/(eta + torch.sum(counts)) and new_label < num_categories):
                # Add a new label
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
            else:
                # Select an existing label
                new_label = Categorical(counts).sample().item()
                labels = torch.cat((labels, torch.tensor([new_label])), dim=0)
        return labels
    
    def generate_nCRP(self, sample_size: int, eta: float):
        '''
        Generate a nested Chinese Restaurant Process with sample size sample_size and concentration parameter eta
        
        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - eta (float): the concentration parameter of the Chinese Restaurant Process

        Returns:
        - label_hierarchy (torch.Tensor): the labels of the samples in the nested Chinese Restaurant Process
        '''
        label_hierarchy = []
        labels = self.generate_CRP(sample_size, eta)
        label_hierarchy.append(labels)
        for l in range(self.layers-1):
            unique_values, counts = self.summarize_CRP(labels)
            num_categories = unique_values.shape[0]
            with Pool(num_categories) as p:
                params =list(itertools.product(counts.tolist(), [eta]))
                labels = p.starmap(self.generate_CRP, params)
            label_hierarchy.append(torch.cat(labels, dim=0))
        return torch.stack(label_hierarchy, dim=0).t()

    def generate_HDP(self):
        '''
        Generate a Hierarchical Dirichlet Process
        '''
        gamma = Gamma(1, 1).sample()
        Global  = DirichletProcess(gamma)
        HDP_distributions = []
        for l in range(self.layers):
            alpha = Gamma(1, 1).sample()
            Local = DirichletProcess(alpha, Global.get_distribution())
            Global.sample(1)
            Local.sample(1)
            print("Global values: ", Global.get_values())
            print("Global weights: ", Global.get_weights())
            print("Local values: ", Local.get_values())
            print("Local weights: ", Local.get_weights())



if __name__ == "__main__":
    # dp = DirichletProcess(1)
    # dp.sample(100)
    # print(dp.get_values())
    # print(dp.get_weights())

    hp = HierarchicalDirichletProcess(2)
    labels = hp.generate_nCRP(100, 1)
    print(labels)
