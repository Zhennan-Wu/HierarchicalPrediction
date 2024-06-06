import torch
import pyro
import itertools
import jax
import jax.numpy as jnp

from pyro.distributions import Dirichlet, Gamma, Categorical
from torch.multiprocessing import Pool
from typing import Union


class Categorical_Distribution:
    def __init__(self, weights: torch.Tensor, values: torch.Tensor):
        '''
        Initialize a Categorical Distribution with weights

        Parameters:
        - weights (list): the weights of the values
        - values (list): the values of the Categorical Distribution
        '''
        self.weights = weights
        self.values = values
    
    def sample(self):
        '''
        Sample from the Categorical Distribution 
        '''
        idx = Categorical(self.weights).sample()
        return self.values[idx.item()]
    

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
            else:
                # Sample from the base distribution
                new_entry = self.base_distribution.sample()
                unseen = True
                for index, entry in enumerate(self.values):
                    if (torch.equal(entry, new_entry)):
                        self.weights[index] += 1
                        unseen = False
                        break
                if (unseen):
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
        return {"values": torch.stack(self.values), "weights": torch.tensor(self.weights)}


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
        - parent_categories_counts (torch.Tensor): the number of samples in each parent category
        - eta (float): the concentration parameter of the Chinese Restaurant Process
        - num_categories (int): the number of child categories to generate

        Returns:
        - labels (torch.Tensor): the labels of the samples
        '''
        num_parent_categories = parent_categories_counts.shape[0]
        if (num_categories < num_parent_categories):
            raise ValueError("The number of child categories should be greater than the number of parent categories")
        # Randomly assign child categories to parent categories
        child_categories = list(range(num_categories))
        child_categories = [[x] for x in child_categories]
        parent_child_relation = dict(zip(list(range(num_categories)),child_categories))
        additional_categories = num_categories - num_parent_categories
        for i in range(additional_categories):
            parent_category = torch.randint(0, num_parent_categories, (1,))
            parent_child_relation[parent_category.item()].append(i+num_parent_categories)
        parent_categories_counts = parent_categories_counts.to(torch.int).tolist()
        parent_child_list = list(parent_child_relation.values())

        # Generate instances based on the assigned parent-child categories
        labels = []
        for p_count, p_c in zip(parent_categories_counts, parent_child_list):
            # Generate the first instance under the parent category
            p_labels = []
            candidates = list(set(p_c) - set(p_labels))
            new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
            p_labels.append(new_label)      
            # Chinese resutaurant process to generate the rest of the instances      
            for _ in range(p_count-1):
                counts = len(set(p_labels))
                candidates = list(set(p_c) - set(p_labels))
                if (torch.rand(1) < eta/(eta + counts) and len(candidates) > 0):
                    # Add a new label
                    new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    p_labels.append(new_label)
                else:
                    unique_values, counts = torch.unique(torch.tensor(p_labels), return_counts=True)
                    new_label_index = Categorical(counts).sample().item()
                    new_label = unique_values[new_label_index].item()
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
            num_categories_per_layer[l] = unique_values.shape[0]

        hierarchy_tree = {}
        for entry in label_hierarchy:
            level = hierarchy_tree
            for l, category in enumerate(entry):
                if category.item() not in level:
                    if (l == self.layers - 1):
                        level[category.item()] = 1
                    else:
                        level[category.item()] = {}
                else:
                    if (l == self.layers - 1):
                        level[category.item()] += 1
                level = level[category.item()]

        return num_categories_per_layer, hierarchy_tree
    
    def _extract_child_layer(self, parent_layer: list):
        '''
        Extract the child layers from the parent layers

        Parameters:
        - parent_layer (list): the parent layer of list of hierarchical dictionaries

        Returns:    
        - child_layer (list): the child layer of list of hierarchical dictionaries
        '''
        child_layer = []
        counts = []
        child_sample_sizes = []
        for parent in parent_layer:
            child = list(parent.values())
            sample_size = []
            for c in child:
                leaves = jax.tree_util.tree_leaves(c)
                sample_size.append(sum(leaves))
            child_sample_sizes = child_sample_sizes + sample_size
            count = len(child)
            child_layer = child_layer + child
            counts.append(count)
        return child_layer, child_sample_sizes, counts

    def update_hierarchy_dict(self, Distributions: list, counts: list, parent_hierarchy: dict = None):
        '''
        Update the hierarchy tree with the distributions
        '''
        child_dict = {}
        if (parent_hierarchy is None):
            child_keys = list(range(len(Distributions)))
            child_dict = dict(zip(child_keys, Distributions))
        else:
            parent_keys = list(parent_hierarchy.keys())
            if (len(parent_keys) != len(counts)):
                raise ValueError("The number of parent keys {} should be equal to the number of counts {}".format(len(parent_keys), len(counts)))
            child_keys = []
            for key, count in zip(parent_keys, counts):
                child_key =  []
                for c in range(count):
                    child_key.append(str(key) + str(c))
                child_keys = child_keys + child_key
            if (len(child_keys) != len(Distributions)):
                raise ValueError("The number of child keys {} should be equal to the number of Distributions {}".format(len(child_keys), len(Distributions)))
            child_dict = dict(zip(child_keys, Distributions))
        return child_dict

    def generate_HDP(self, sample_size: int, hierarchy_tree: dict):
        '''
        Generate a Hierarchical Dirichlet Process
        '''
        gamma = Gamma(1, 1).sample()
        Global = DirichletProcess(gamma, sample_size) 
        HDP_structure = []
        HDP_distributions = []
        HDP_sample_sizes = []
        counts = [len(list(hierarchy_tree.keys()))]
        HDP_distributions.append([Global.get_distribution()]*counts[0])
        sample_sizes = []
        for c in list(hierarchy_tree.values()):
            leaves = jax.tree_util.tree_leaves(c)
            sample_sizes.append(sum(leaves))
        HDP_sample_sizes.append(sample_sizes)
        level = list(hierarchy_tree.values())
        for l in range(self.layers):
            alpha = Gamma(1, 1).sample()
            base = HDP_distributions[-1]
            base_sample_sizes = HDP_sample_sizes[-1]
            alpha_list = [alpha.item()]*len(base_sample_sizes)
            param = list(zip(alpha_list, base_sample_sizes, base))
            with Pool(len(base)) as p:
                DPs = p.starmap(DirichletProcess, param)
            if (l == 0):
                HDP_structure.append(self.update_hierarchy_dict(DPs, counts))
            else:
                HDP_structure.append(self.update_hierarchy_dict(DPs, counts, HDP_structure[-1]))
            print("HDP_structure")
            print(HDP_structure)
            if (l < self.layers - 1):
                level, sample_sizes, counts = self._extract_child_layer(level)
                child_distributions = []
                if (len(counts) != len(DPs)):
                    raise ValueError("The number of child layers {} should be equal to the number of Dirichlet Processes {}".format(len(counts), len(DPs)))
                for count, DP in zip(counts, DPs):
                    next_base = [DP.get_distribution()]*count
                    child_distributions = child_distributions + next_base
                HDP_distributions.append(child_distributions)
                HDP_sample_sizes.append(sample_sizes)
            else:
                sample_sizes = level
                counts = [1]*len(level)
                child_distributions = []
                if (len(counts) != len(DPs)):
                    raise ValueError("The number of child layers {} should be equal to the number of Dirichlet Processes {}".format(len(counts), len(DPs)))
                for count, DP in zip(counts, DPs):
                    next_base = [DP.get_distribution()]*count
                    child_distributions = child_distributions + next_base
                HDP_distributions.append(child_distributions)
                HDP_sample_sizes.append(sample_sizes)

        return HDP_distributions

    def visualize_HDP(self, HDP_distributions: list, labels: torch.Tensor):
        '''
        Visualize the Hierarchical Dirichlet Process
        '''
        for l, distributions in enumerate(HDP_distributions):
            print("Layer: ", l)
            print("Number of subclass: ", len(distributions))
            print("Number of ")
            for d in distributions:
                print("Values: ", d["values"])
                print("Weights: ", d["weights"])

    def infer_HDP(self, HDP_distributions: list):
        '''
        Infer the Hierarchical Dirichlet Process
        '''
        pass
    
    def match_HDP(self, HDP_distributions: list, hierarchy_tree: dict):
        '''
        Match the Hierarchical Dirichlet Process
        '''
        pass


if __name__ == "__main__":
    # dp = DirichletProcess(1)
    # dp.sample(100)
    # print(dp.get_values())
    # print(dp.get_weights())

    hp = HierarchicalDirichletProcess(3, {2: 4})
    labels = hp.generate_nCRP(50, 1)
    print("labels")
    print(labels)
    num_categories_per_layer, hierarchy_tree = hp.summarize_nCRP(labels)
    print("num_categories_per_layer")
    print(num_categories_per_layer)
    print("hierarchy_tree")
    print(hierarchy_tree)
    hdp = hp.generate_HDP(100, hierarchy_tree)
    hp.visualize_HDP(hdp, labels)
