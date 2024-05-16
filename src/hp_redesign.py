import torch
import pyro
import itertools

from pyro.distributions import Dirichlet, Gamma, Categorical
from torch.multiprocessing import Pool
from typing import Union


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
    def __init__(self, alpha: float, sample_size: int, base_distribution: dict = None):
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
        
        self.sample(sample_size)
    
    def sample(self, num_samples: int):
        '''
        Sample from the Dirichlet Process

        Parameters:
        - num_samples (int): the number of samples to draw
        '''
        # Sample from the base distribution with probability alpha / (alpha + N)
        for _ in range(num_samples):
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
    def __init__(self, layers: int, fixed_layers: dict = None):
        '''
        Initialize a Hierarchical Dirichlet Process with layers

        Parameters:
        - layers (int): the number of layers in the Hierarchical Dirichlet Process
        '''
        self.layers = layers
        self.layer_constrains = False
        self.implied_constraints = None
        self.fixed_layers = fixed_layers
        if (fixed_layers is not None):
            fix_keys = list(fixed_layers.keys())
            fix_values = list(fixed_layers.values())
            if (len(fix_keys) > 1):
                if (all(fix_keys[i] <= fix_keys[i+1] for i in range(len(fix_keys)-1))):
                    raise ValueError("The fixed layers should be in increasing order, get {}".format(fix_keys))
                if (all(fix_values[i] <= fix_values[i+1] for i in range(len(fix_values)-1))):
                    raise ValueError("The fixed layers should have increasing number of categories, get{}".format(fix_values))
            self.layer_constrains = True
            layer_index = [float('inf')]*layers
            self.implied_constraints = {}
            for i in range(layers):
                if (i in fixed_layers.keys()):
                    layer_index[i] = fixed_layers[i]
            for i in range(layers):
                self.implied_constraints[i] = min(layer_index[i:])
        self.category_hierarchy = []

    def summarize_CRP(self, labels: Union[torch.Tensor, list], indices: Union[torch.Tensor, list]):
        '''
        Summarize the Chinese Restaurant Process to get the unique values and their counts

        Parameters:
        - labels (torch.Tensor or list): the labels of the samples

        Returns:
        - unique_values (torch.Tensor): the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        - counts (torch.Tensor): the counts of the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        '''
        unique_values_list = []
        label_indices_list = []
        counts_list = []
        if (isinstance(labels, list)):
            p_cat = 0
            category_dict = {}
            for label, index in zip(labels, indices):
                unique_values, inverse_indices, counts = torch.unique(label, return_inverse = True, return_counts=True)
                category_dict[p_cat] = {val.item(): {} for val in unique_values}
                p_cat += 1
                for i in range(unique_values.shape[0]):
                    label_indices_list.append(index[torch.where(inverse_indices == i)])
                unique_values_list.append(unique_values)
                counts_list.append(counts)
            unique_values = torch.cat(unique_values_list, dim=0)
            counts = torch.cat(counts_list, dim=0)
            self.category_hierarchy.append(category_dict)
        else:
            unique_values, inverse_indices, counts = torch.unique(labels, return_inverse = True, return_counts=True)
            self.category_hierarchy.append({val.item(): {} for val in unique_values})
            for i in range(unique_values.shape[0]):
                label_indices_list.append(indices[torch.where(inverse_indices == i)])
        return unique_values, label_indices_list, counts
    
    def generate_CRP(self, sample_size: int, eta: float):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta
        
        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - eta (float): the concentration parameter of the Chinese Restaurant Process

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        labels = torch.tensor([0], dtype=torch.int32)
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

    def generate_fixed_categories(self, parent_categories_counts: torch.Tensor, eta: float, num_categories: int):
        '''
        Generate a Chinese Restaurant Process with sample size sample_size and concentration parameter eta with fixed number of categories

        Parameters:
        - sample_size (int): the number of samples to generate labels for
        - eta (float): the concentration parameter of the Chinese Restaurant Process
        - num_categories (int): the number of categories to generate

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        num_parent_categories = parent_categories_counts.shape[0]
        if (num_categories < num_parent_categories):
            raise ValueError("The number of child categories should be greater than the number of parent categories")
        child_categories = list(range(num_categories))
        child_categories = [[x] for x in child_categories]
        parent_child_relation = dict(zip(list(range(num_categories)),child_categories))
        additional_categories = num_categories - num_parent_categories
        for i in range(additional_categories):
            parent_category = torch.randint(0, num_parent_categories, (1,))
            parent_child_relation[parent_category.item()].append(i+num_parent_categories)
        parent_categories_counts = parent_categories_counts.to(torch.int).tolist()
        parent_child_list = list(parent_child_relation.values())

        labels = []
        for p_count, p_c in zip(parent_categories_counts, parent_child_list):
            p_labels = []
            for _ in range(p_count):
                counts = len(set(p_labels))
                candidates = list(set(p_c) - set(p_labels))
                if (torch.rand(1) < eta/(eta + counts) and len(candidates) > 0):
                    # Add a new label
                    new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    p_labels.append(new_label)
                else:
                    # Select an existing label
                    new_label = p_labels[torch.randint(0, len(p_labels), (1,)).item()]
                    p_labels.append(new_label)
            labels.append(torch.tensor(p_labels))

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
        prev_labels = self.generate_CRP(sample_size, eta)
        prev_indices = torch.arange(sample_size)
        prev_unique_values = torch.zeros(1)
        prev_counts = torch.tensor([sample_size])
        label_hierarchy.append(prev_labels)
        l = 1
        while (l < self.layers):
            unique_values, indices, counts = self.summarize_CRP(prev_labels, prev_indices)
            num_categories = unique_values.shape[0]
            if (num_categories > self.implied_constraints[l]):
                label_hierarchy.pop()
                if (l == 1):
                    indices = torch.arange(sample_size).unsqueeze(0)
                    unique_values = torch.zeros(1).unsqueeze(0)
                    counts = torch.tensor(sample_size).unsqueeze(0)
                    num_categories = 1
                else:
                    indices = prev_indices
                    unique_values = prev_unique_values
                    counts = prev_counts
                    num_categories = unique_values.shape[0]
                num_subcategories = self.implied_constraints[l]
                labels =self.generate_fixed_categories(counts, eta, num_subcategories)   
                l -= 1
            elif (l in list(self.fixed_layers.keys())):
                num_subcategories = self.fixed_layers[l]
                labels =self.generate_fixed_categories(counts, eta, num_subcategories)               
            else:
                with Pool(num_categories) as p:
                    params =list(itertools.product(counts.tolist(), [eta]))
                    labels = p.starmap(self.generate_CRP, params)
            if (isinstance(indices, list)):
                global_indices = torch.cat(indices, dim=0)
            else:
                global_indices = indices
            new_layer_label = torch.zeros(sample_size, dtype=torch.long)
            new_layer_label[global_indices] = torch.cat(labels, dim=0)
            label_hierarchy.append(new_layer_label)

            prev_labels = labels
            prev_indices = indices
            prev_unique_values = unique_values
            prev_counts = counts
            l += 1

        return torch.stack(label_hierarchy, dim=0).t()

    def summarize_nCRP(self, label_hierarchy: torch.Tensor):
        '''
        Summarize the nested Chinese Restaurant Process to get the unique values and their counts

        Parameters:
        - label_hierarchy (torch.Tensor): the labels of the samples in the nested Chinese Restaurant Process

        Returns:
        - unique_values (torch.Tensor): the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        - counts (torch.Tensor): the counts of the unique values in the labels tensor or list of tensors (same value in different tensors are considered as different values)
        '''

        num_categories_per_layer = {}
        for l in range(self.layers):
            unique_values = torch.unique(label_hierarchy[:, :l+1], dim=0)
            print("layer: ", l)
            print("unique values: ", unique_values)
            num_categories_per_layer[l] = unique_values.shape[0]
        unique_values = torch.unique(label_hierarchy, dim=0)
        return num_categories_per_layer
    
    
    def generate_HDP(self, sample_size: int):
        '''
        Generate a Hierarchical Dirichlet Process
        '''
        gamma = Gamma(1, 1).sample()
        Global = DirichletProcess(gamma, sample_size)
        Global.sample(sample_size)
        HDP_distributions = []
        HDP_distributions.append([Global])
        for l in range(self.layers):
            alpha = Gamma(1, 1).sample()
            base = HDP_distributions[-1]
            param = list(itertools.product([alpha], [sample_size], base))
            with Pool(len(base)) as p:
                DPs = p.starmap(DirichletProcess, param)
            child_distributions = [DP.get_distribution() for DP in DPs]
            HDP_distributions.append(child_distributions)


if __name__ == "__main__":
    # dp = DirichletProcess(1)
    # dp.sample(100)
    # print(dp.get_values())
    # print(dp.get_weights())

    hp = HierarchicalDirichletProcess(3, {2: 4})
    labels = hp.generate_nCRP(100, 1)
    print(labels)
    num_categories_per_layer = hp.summarize_nCRP(labels)
    print(num_categories_per_layer)
