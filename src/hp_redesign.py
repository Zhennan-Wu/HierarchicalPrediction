#############################################################################################################
# To Do:
# - Implement parallel Gibbs sampling for inference
# 
import torch
import pyro
import itertools
import jax
import jax.numpy as jnp
import copy
import random

from pyro.distributions import Dirichlet, Gamma, Categorical
from torch.multiprocessing import Pool
from typing import Any, Union, List, Tuple, Dict
from jax.tree_util import PyTreeDef
from utils import TreeNode, Tree

PyTree = Union[jnp.ndarray, List['PyTree'], Tuple['PyTree', ...], Dict[Any, 'PyTree']]


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
    def __init__(self, alpha: float, sample_size: int, base_distribution: dict):
        '''
        Initialize a Dirichlet Process with concentration parameter alpha

        Parameters:
        - alpha (float): the concentration parameter of the Dirichlet Process
        - base_distribution (Dict): the base distribution of the Dirichlet Process
        '''
        self.alpha = alpha
        self.values = []
        self.weights = []
        self.base_distribution = Categorical_Distribution(base_distribution["weights"], base_distribution["values"])
        pyro.set_rng_seed(sample_size + random.randint(0, 1000))
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
   
   
def calc_sequential_stick_breaking_weight(alpha: float, parent_weights: list, num_categories: int):
    '''
    Generate the hierarchical distributions
    '''
    child_weights = []
    v_values = []
    concentrate1 = alpha
    for k in range(num_categories):
        concentrate0 = alpha*parent_weights[k]
        concentrate1 -= concentrate0
        concentrate0 = max(concentrate0, 1e-3)
        concentrate1 = max(concentrate1, 1e-3)
        v_prime = torch.distributions.Beta(concentrate0, concentrate1).sample()
        v_values.append(v_prime)
        pi_final = v_prime
        for j in range(k):
            pi_final *= 1 - v_values[j]
        child_weights.append(pi_final)
    return child_weights


def transfer_index_tensor_to_string(index: torch.Tensor):
    '''
    Transfer the index tensor to string
    '''
    index_list = index.tolist()
    index_str = ''.join([str(i) for i in index_list])
    return index_str


class HierarchicalDirichletProcess:
    def __init__(self, latent_dimension: int, layers: int, batch_size: int, truncated_length: int, fixed_layers: dict = None):
        '''
        Initialize a Hierarchical Dirichlet Process with layers

        Parameters:
        - num_of_words (int): the number of words in the vocabulary
        - layers (int): the number of layers in the Hierarchical Dirichlet Process
        - fixed_layers (dict): the fixed number of categories in each layer
        - global_sample_size (int): the number of samples to draw from the Global Dirichlet Process
        '''
        self.hyperparameters = {}
        self.batch_size = batch_size
        self.truncate_length = truncated_length
        self.latent_dimension = latent_dimension
        self.layers = layers
        self.layer_constrains = False
        self.implied_constraints = None
        self.fixed_layers = fixed_layers
        self._check_layer_constraints()

        # a list of dictionaries, to record the number of subcategories in each layer
        self.number_of_subcategories = [] 


        # Initialize the base distribution
        beta = Gamma(1, 1).sample()
        self.hyperparameters["BASE"] = beta
        self.parameters = self.generate_parameters()

        # Initialize hierarchical structure
        eta = Gamma(1, 1).sample()
        self.hyperparameters["CRF"] = eta
        self.labels = self.generate_CRF()
        self.get_number_of_subcategories()
        # self.hdp = self.generate_HDP(batch_size, self.hierarchy_tree, self.labels)

        # Record hierarchical distributions
        beta = Gamma(1, 1).sample()
        self.hyperparameters["GLOBAL"] = beta
        self.base_weight = self.generate_base_weights()
        
        self.distributions = []
        self.hyperparameters["DP"] = {}
        self.generate_hierarchy_tree()

    def print_hyperparameters(self):
        '''
        Print the hyperparameters of the Hierarchical Dirichlet Process
        '''
        print(self.hyperparameters)
    
    def print_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        print(self.distributions)
    
    def print_base_weights(self):
        '''
        Print the base weights of the Hierarchical Dirichlet Process
        '''
        print(self.base_weight)
    
    def print_labels(self):
        '''
        Print the labels of the Hierarchical Dirichlet Process
        '''
        print(self.labels)
    
    def print_number_of_subcategories(self):
        '''
        Print the number of subcategories in each layer of the Hierarchical Dirichlet Process
        '''
        print(self.number_of_subcategories)
    
    def print_parameters(self):
        '''
        Print the parameters of the Hierarchical Dirichlet Process
        '''
        print(self.parameters)

    def generate_parameters(self):
        '''
        Generate the parameters for the Hierarchical Dirichlet Process

        Returns:
        - parameters (torch.Tensor): the distribution parameters of the Hierarchical Dirichlet Process
        '''
        beta = self.hyperparameters["BASE"]
        base_distribution = Dirichlet(beta * torch.ones(self.latent_dimension))
        return base_distribution.sample((self.truncate_length, ))

    def generate_base_weights(self):
        '''
        Generate the base weights of the Hierarchical Dirichlet Process
        '''
        beta = self.hyperparameters["GLOBAL"]
        remaining_weight = 1
        weights = []
        for _ in range(self.truncate_length):
            pi_prime = torch.distributions.Beta(1, beta).sample()
            pi_value = pi_prime * remaining_weight
            weights.append(pi_value)
            remaining_weight *= (1 - pi_prime)
           
        if (sum(weights) > 1):
            raise ValueError("The sum of the weights should be smaller than 1, instead got {}".format(sum(weights)))
        return weights
    
    def generate_CRF(self):
        '''
        Generate a nested Chinese Restaurant Process with sample size sample_size and concentration parameter eta

        Returns:
        - label_hierarchy (torch.Tensor): the labels of the samples in the nested Chinese Restaurant Process
        '''
        eta = self.hyperparameters["CRF"]
        label_hierarchy = []
        parent_labels = self._generate_CRP(self.batch_size, eta)
        indices_group_by_categories = torch.arange(self.batch_size)
        parent_categories = torch.zeros(1)
        parent_counts = torch.tensor([self.batch_size])
        label_hierarchy.append(parent_labels)
        l = 1
        while (l < self.layers):
            categories, indices, counts = self._get_category_info(parent_labels, indices_group_by_categories)
            num_categories = categories.shape[0]
            if (num_categories > self.implied_constraints[l]):
                label_hierarchy.pop()
                if (l == 1):
                    indices = torch.arange(self.batch_size).unsqueeze(0)
                    categories = torch.zeros(1).unsqueeze(0)
                    counts = torch.tensor(self.batch_size).unsqueeze(0)
                    num_categories = 1
                else:
                    indices = indices_group_by_categories
                    categories = parent_categories
                    counts = parent_counts
                    num_categories = categories.shape[0]
                num_subcategories = self.implied_constraints[l]
                labels =self._generate_fixed_categories(counts, eta, num_subcategories)   
                l -= 1
            elif (l in list(self.fixed_layers.keys())):
                num_subcategories = self.fixed_layers[l]
                labels =self._generate_fixed_categories(counts, eta, num_subcategories)               
            else:
                with Pool(num_categories) as p:
                    params =list(itertools.product(counts.tolist(), [eta]))
                    labels = p.starmap(self._generate_CRP, params)
            if (isinstance(indices, list)):
                global_indices = torch.cat(indices, dim=0)
            else:
                global_indices = indices
            new_layer_label = torch.zeros(self.batch_size, dtype=torch.int)
            new_layer_label[global_indices] = torch.cat(labels, dim=0).int()
            label_hierarchy.append(new_layer_label)

            parent_labels = labels
            indices_group_by_categories = indices
            parent_categories = categories
            parent_counts = counts
            l += 1

        return torch.stack(label_hierarchy, dim=0).t()

    def get_number_of_subcategories(self):
        '''
        Get the number of subcategories in the Hierarchical Dirichlet Process
        '''
        label_hierarchy = self.labels

        for l in range(self.layers):
            child_categories = torch.unique(label_hierarchy[:, :l+2], dim=0)
            parent_categories, number_of_children = torch.unique(child_categories[:, :-1], dim=0, return_counts=True)
            
            num_subcategories = {}
            for pc, nc in zip(parent_categories, number_of_children):
                index_string = transfer_index_tensor_to_string(pc)
                num_subcategories[index_string] = nc.item()   
            self.number_of_subcategories.append(num_subcategories)

        parent_categories = torch.unique(label_hierarchy, dim=0)
        num_subcategories = {}
        for pc in parent_categories:
            index_string = transfer_index_tensor_to_string(pc)
            num_subcategories[index_string] = 1   
        self.number_of_subcategories.append(num_subcategories)
    
    def generate_hierarchy_tree(self):
        '''
        Generate the hierarchical tree from the label hierarchy
        '''
        # First level
        child_categories = self.number_of_subcategories[0].keys()
        etas = Gamma(1, 1).sample((len(child_categories),)).tolist()
        hyper_params = dict(zip(child_categories, etas))
        self.hyperparameters["DP"].update(hyper_params)

        with Pool(len(child_categories)) as p:
            weights = [self.base_weight]*len(child_categories)
            truncated_lengths = [self.truncate_length]*len(child_categories)
            params = list(zip(etas, weights, truncated_lengths))
            distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
        self.distributions.append(dict(zip(child_categories, distributions)))

        for l in range(self.layers):
            parents = self.number_of_subcategories[l].keys()
            num_childs = self.number_of_subcategories[l].values()
            children = self.number_of_subcategories[l+1].keys()
            total_num_childs = sum(num_childs)
            etas = Gamma(1, 1).sample((total_num_childs,)).tolist()
            hyper_params = dict(zip(children, etas))
            self.hyperparameters["DP"].update(hyper_params)

            truncated_lengths = [self.truncate_length]*total_num_childs
            parents_weights = []
            for parent, nc in zip(parents, num_childs):
                parents_weights += [self.distributions[l][parent]]*nc
            
            with Pool(total_num_childs) as p:
                params = list(zip(etas, parents_weights, truncated_lengths))
                distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
            self.distributions.append(dict(zip(children, distributions)))

    def _check_layer_constraints(self):
        '''
        Check if the layer constraints are satisfied
        '''
        if (self.fixed_layers is not None):
            fix_keys = list(self.fixed_layers.keys())
            fix_values = list(self.fixed_layers.values())
            if (len(fix_keys) > 1):
                if (all(fix_keys[i] <= fix_keys[i+1] for i in range(len(fix_keys)-1))):
                    raise ValueError("The fixed layers should be in increasing order, get {}".format(fix_keys))
                if (all(fix_values[i] <= fix_values[i+1] for i in range(len(fix_values)-1))):
                    raise ValueError("The fixed layers should have increasing number of categories, get {}".format(fix_values))
            self.layer_constrains = True
            layer_index = [float('inf')]*self.layers
            self.implied_constraints = {}
            for i in range(self.layers):
                if (i in self.fixed_layers.keys()):
                    layer_index[i] = self.fixed_layers[i]
            for i in range(self.layers):
                self.implied_constraints[i] = min(layer_index[i:])

    # def filter_related_samples(self, sample_index: int, samples: torch.Tensor, hierarchy_params_tree: dict, filter_type: str):
    #     '''
    #     Filter the related samples based on the sample index and the hierarchical parameters tree
    #     '''
    #     if (filter_type == "sample"):
    #         pass
    #     elif (filter_type == "table"):
    #         pass
    #     else:
    #         raise ValueError("The filter type {} is not supported".format(filter_type))
    #     pass

    # def calc_conditional_density(self, samples: torch.Tensor, mixture_index: int, sample_index: int, hierarchy_params_tree: dict):
    #     '''
    #     Calculate the conditional density of the Hierarchical Dirichlet Process
    #     '''
    #     prior = self.global_dist["weights"].view(-1, 1)/torch.sum(self.global_dist["weights"]) 

    #     related_samples = self.filter_related_samples(sample_index, mixture_index, samples, hierarchy_params_tree, "sample")
    #     all_related_samples = torch.transpose(torch.cat((related_samples, samples[sample_index].view(1, -1)), dim=0), 0, 1)
    #     related_samples = torch.transpose(related_samples, 0, 1)
    #     # calculate denominator
    #     log_joint_prob = torch.log(torch.inner(self.global_dist["values"]*prior, related_samples)) # dimension: number of parameter samples * number of samples
    #     joint_prob = torch.exp(torch.sum(log_joint_prob, dim=-1))
    #     likelihood_denominator = torch.sum(joint_prob)

    #     # calculate nominator
    #     log_joint_prob = torch.log(torch.inner(self.global_dist["values"]*prior, all_related_samples)) # dimension: number of parameter samples * number of samples
    #     joint_prob = torch.exp(torch.sum(log_joint_prob, dim=-1))
    #     likelihood_nominator = torch.sum(joint_prob)
    #     cdf = likelihood_nominator/likelihood_denominator
    #     return cdf
    
    # def get_sample_index_from_table_index(self, table_index: int, hierarchy_params_tree: dict):
    #     '''
    #     Get the sample index from the table index
    #     '''
    #     pass

    # def calc_table_conditional_density(self, samples: torch.Tensor, mixture_index: int, table_index: int, hierarchy_params_tree: dict):
    #     '''
    #     Calculate the conditional density of the Hierarchical Dirichlet Process
    #     '''
    #     prior = self.global_dist["weights"].view(-1, 1)/torch.sum(self.global_dist["weights"]) 

    #     related_samples = self.filter_related_samples(table_index, mixture_index, samples, hierarchy_params_tree, "table")
    #     sample_index = self.get_sample_index_from_table_index(table_index, hierarchy_params_tree)
    #     all_related_samples = torch.transpose(torch.cat((related_samples, samples[sample_index]), dim=0), 0, 1)
    #     related_samples = torch.transpose(related_samples, 0, 1)
    #     # calculate denominator
    #     log_joint_prob = torch.log(torch.inner(self.global_dist["values"]*prior, related_samples)) # dimension: number of parameter samples * number of samples
    #     joint_prob = torch.exp(torch.sum(log_joint_prob, dim=-1))
    #     likelihood_denominator = torch.sum(joint_prob)

    #     # calculate nominator
    #     log_joint_prob = torch.log(torch.inner(self.global_dist["values"]*prior, all_related_samples)) # dimension: number of parameter samples * number of samples
    #     joint_prob = torch.exp(torch.sum(log_joint_prob, dim=-1))
    #     likelihood_nominator = torch.sum(joint_prob)
    #     cdf = likelihood_nominator/likelihood_denominator
    #     return cdf

    # def calculate_posterior(self, samples: torch.Tensor, hierarchy_params_tree: dict, K: int, layer: int):
    #     '''
    #     Calculate the posterior of the nested Chinese Restaurant Process
    #     '''
    #     if (layer == 1):
    #         global_new_probs = []
    #         f_value = []
    #         for sample in samples:
    #             new_probs = []
    #             for k in range(K):
    #                 new_prob = self.calc_conditional_density(samples, k, sample, hierarchy_params_tree)
    #                 f_value.append(new_prob)
    #                 weight = 1 # remained to be completed
    #                 new_probs.append(new_prob*weight)
    #             global_new_probs.append(new_probs)
    #         count = hierarchy_param_tree[0] # remained to be completed
    #         posterior = torch.stack(f_value*count, global_new_probs)
    #     elif (layer == 2):
    #         global_new_probs = []
    #         f_value = []
    #         for sample in samples:
    #             new_probs = []
    #             for k in range(K):
    #                 new_prob = self.calc_table_conditional_density(samples, k, sample, hierarchy_params_tree)
    #                 f_value.append(new_prob)
    #                 weight = 1
    #                 new_probs.append(new_prob*weight)
    #             global_new_probs.append(new_probs)
    #         count = hierarchy_param_tree[0]
    #         posterior = torch.stack(f_value*count, global_new_probs)
    #     return posterior

    def _get_category_info(self, labels: Union[torch.Tensor, list], indices: Union[torch.Tensor, list]):
        '''
        Summarize the Chinese Restaurant Process to get the unique values and their counts

        Parameters:
        - labels (torch.Tensor or list): the labels of the samples
        - indices (torch.Tensor or list): the indices of the samples that belongs to each parent category

        Returns:
        - categories (torch.Tensor): all different categories among different parent categories
        - samples_group_by_category (list): a list contains the indices of the samples that belongs to each parent category
        - counts_by_category (torch.Tensor): the counts of the samples that belongs to each category
        '''
        category_list = []
        samples_group_by_category = []
        counts_by_category_list = []
        if (isinstance(labels, list)):
            for label, index in zip(labels, indices):
                categories, sample_category_assignments, counts_by_category = torch.unique(label, return_inverse = True, return_counts=True)
                for i in range(categories.shape[0]):
                    samples_group_by_category.append(index[torch.where(sample_category_assignments == i)])
                category_list.append(categories)
                counts_by_category_list.append(counts_by_category)
            categories = torch.cat(category_list, dim=0)
            counts_by_category = torch.cat(counts_by_category_list, dim=0)
        else:
            categories, sample_category_assignments, counts_by_category = torch.unique(labels, return_inverse = True, return_counts=True)
            for i in range(categories.shape[0]):
                samples_group_by_category.append(indices[torch.where(sample_category_assignments == i)])
        return categories, samples_group_by_category, counts_by_category
    
    def _generate_CRP(self, sample_size: int, eta: float):
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
            categories, counts = torch.unique(labels, return_counts=True)
            if (torch.rand(1) < eta/(eta + torch.sum(counts))):
                # Add a new label
                new_label = torch.max(categories) + 1
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
            else:
                # Select an existing label
                index = Categorical(counts).sample()
                new_label = categories[index.int()]
                labels = torch.cat((labels, new_label.unsqueeze(0)), dim=0)
        return labels

    def _generate_fixed_categories(self, parent_categories_counts: torch.Tensor, eta: float, num_categories: int):
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
            p_labels_record = []
            candidates = p_c
            new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
            p_labels_record.append(new_label) 
            p_labels.append(0)    
            # Chinese resutaurant process to generate the rest of the instances      
            for _ in range(p_count-1):
                counts = len(set(p_labels_record))
                candidates = list(set(p_c) - set(p_labels_record))
                if (torch.rand(1) < eta/(eta + counts) and len(candidates) > 0):
                    # Add a new label
                    new_label = candidates[torch.randint(0, len(candidates), (1,)).item()]
                    p_labels_record.append(new_label)
                    p_labels.append(max(p_labels)+1)
                else:
                    categories, counts = torch.unique(torch.tensor(p_labels), return_counts=True)
                    new_label_index = Categorical(counts).sample().item()
                    new_label = categories[new_label_index].item()
                    p_labels.append(new_label)
            labels.append(torch.tensor(p_labels))
        return labels
    
    # def _extract_child_layer(self, parent_layer: list):
    #     '''
    #     Extract the child layers from the parent layers

    #     Parameters:
    #     - parent_layer (list): the parent layer of list of hierarchical dictionaries

    #     Returns:    
    #     - child_layer (list): the child layer of list of hierarchical dictionaries
    #     '''
    #     child_layer = []
    #     counts = []
    #     child_sample_sizes = []
    #     for parent in parent_layer:
    #         child = list(parent.values())
    #         sample_size = []
    #         for c in child:
    #             leaves = jax.tree_util.tree_leaves(c)
    #             sample_size.append(sum(leaves))
    #         child_sample_sizes = child_sample_sizes + sample_size
    #         count = len(child)
    #         child_layer = child_layer + child
    #         counts.append(count)
    #     return child_layer, child_sample_sizes, counts

    # def update_hierarchy_dict(self, Distributions: list, counts: list, labels: torch.Tensor, parent_hierarchy: dict = None):
    #     '''
    #     Update the hierarchy tree with the distributions
    #     '''
    #     child_dict = {}
    #     distribution_params = []
    #     for d in Distributions:
    #         distribution_params.append(d.get_distribution())
    #     if (parent_hierarchy is None):
    #         child_keys = list(range(len(Distributions)))
    #         child_keys = [str(key) for key in child_keys]
    #         child_dict = dict(zip(child_keys, distribution_params))
    #     else:
    #         parent_keys = list(parent_hierarchy.keys())
    #         if (len(parent_keys) != len(counts)):
    #             raise ValueError("The number of parent keys {} should be equal to the number of counts {}".format(len(parent_keys), len(counts)))
    #         child_keys = []
    #         for key, count in zip(parent_keys, counts):
    #             child_key =  []
    #             for c in range(count):
    #                 child_key.append(str(key) + str(c))
    #             child_keys = child_keys + child_key
    #         if (len(child_keys) != len(Distributions)):
    #             raise ValueError("The number of child keys {} should be equal to the number of Distributions {}".format(len(child_keys), len(Distributions)))
    #         child_dict = dict(zip(child_keys, distribution_params))
    #     return child_dict

    # def generate_parameters(self, sample_size: int, hierarchy_tree: dict, eta: float):
    #     '''
    #     Generate the parameters for the Hierarchical Dirichlet Process
    #     '''
    #     gamma = Gamma(1, 1).sample()
    #     global_scale = 10
    #     Global = DirichletProcess(gamma, global_scale*sample_size, self.global_dist) # Generate global pool
    #     categorical_params = Global.get_distribution()
    #     counts, hierarchy = jax.tree.flatten(hierarchy_tree)
    #     counts_array = torch.tensor(counts)
    #     hp_params_array = torch.zeros_like(counts_array)
    #     hp_params = []
    #     for idx, _ in enumerate(counts):
    #         if (random.random() < eta/(eta + torch.dot(hp_params_array, counts_array).item())):
    #             # Add a new label
    #             new_label = categorical_params["values"][Categorical(categorical_params["weights"]).sample().item()]
    #         else:
    #             # Select an existing label
    #             new_label = hp_params[Categorical(torch.tensor(counts[:idx])).sample().item()]    
    #         hp_params.append(new_label)
    #         hp_params_array[idx] = 1       
    #     hierarchy_param_tree = jax.tree.unflatten(hierarchy, hp_params)
    #     return hierarchy_param_tree

    # def generate_HDP(self, sample_size: int, hierarchy_tree: dict, labels: torch.Tensor):
    #     '''
    #     Generate a Hierarchical Dirichlet Process
    #     '''
    #     HDP_distributions_no_duplicate = []
    #     gamma = Gamma(1, 1).sample()
    #     global_scale = 10
    #     Global = DirichletProcess(gamma, global_scale*sample_size, self.global_dist) # Generate global pool
    #     HDP_structure = []
    #     HDP_distributions = []
    #     HDP_sample_sizes = []
    #     counts = [len(list(hierarchy_tree.keys()))]
    #     HDP_distributions.append([Global.get_distribution()]*counts[0])
    #     HDP_distributions_no_duplicate.append([Global.get_distribution()])
    #     sample_sizes = []
    #     for c in list(hierarchy_tree.values()):
    #         leaves = jax.tree_util.tree_leaves(c)
    #         sample_sizes.append(sum(leaves))
    #     HDP_sample_sizes.append(sample_sizes)
    #     level = list(hierarchy_tree.values())
    #     for l in range(self.layers):
    #         alpha = Gamma(1, 1).sample()
    #         base = HDP_distributions[-1]
    #         base_sample_sizes = HDP_sample_sizes[-1] # Get how many samples are in each category
    #         alpha_list = [alpha.item()]*len(base_sample_sizes) # Get the alpha value for each category
    #         param = list(zip(alpha_list, base_sample_sizes, base))
    #         print("Layer: ", l)
    #         print("Param: ", param)
    #         with Pool(len(base)) as p:
    #             DPs = p.starmap(DirichletProcess, param)
    #         if (l == 0):
    #             # Generate the first layer of the HDP
    #             HDP_structure.append(self.update_hierarchy_dict(DPs, counts, labels))
    #         else:
    #             # Generate the rest of the layers of the HDP
    #             HDP_structure.append(self.update_hierarchy_dict(DPs, counts, labels, HDP_structure[-1]))
    #         if (l < self.layers - 1):
    #             level, sample_sizes, counts = self._extract_child_layer(level)
    #             child_distributions = []
    #             child_distributions_no_duplicate = []
    #             if (len(counts) != len(DPs)):
    #                 raise ValueError("The number of child layers {} should be equal to the number of Dirichlet Processes {}".format(len(counts), len(DPs)))
    #             for count, DP in zip(counts, DPs):
    #                 child_distributions_no_duplicate.append(DP.get_distribution())
    #                 next_base = [DP.get_distribution()]*count
    #                 child_distributions = child_distributions + next_base
    #             HDP_distributions.append(child_distributions)
    #             HDP_sample_sizes.append(sample_sizes)
    #             HDP_distributions_no_duplicate.append(child_distributions_no_duplicate)
    #         else:
    #             sample_sizes = level
    #             counts = [1]*len(level)
    #             child_distributions = []
    #             child_distributions_no_duplicate = []
    #             if (len(counts) != len(DPs)):
    #                 raise ValueError("The number of child layers {} should be equal to the number of Dirichlet Processes {}".format(len(counts), len(DPs)))
    #             for count, DP in zip(counts, DPs):
    #                 child_distributions_no_duplicate.append(DP.get_distribution())
    #                 next_base = [DP.get_distribution()]*count
    #                 child_distributions = child_distributions + next_base
    #             HDP_distributions_no_duplicate.append(child_distributions_no_duplicate)
    #             HDP_distributions.append(child_distributions)
    #             HDP_sample_sizes.append(sample_sizes)
    #     self._check_nCRP_HDP_match(labels, HDP_structure)
    #     return HDP_distributions_no_duplicate, HDP_structure

    # def record_active_params(self, top_level_HDP_structure: dict):
    #     '''
    #     Record the active parameters in the Hierarchical Dirichlet Process
    #     '''
    #     for top_class in top_level_HDP_structure.keys():
    #         for value, weight in zip(top_level_HDP_structure[top_class]["values"], top_level_HDP_structure[top_class]["weights"]):
    #             if value in self.activate_params.keys():
    #                 self.activate_params[value] += weight
    #             else:
    #                 self.activate_params[value] = weight

    # def _check_nCRP_HDP_match(self, nCRP_hierarchy: torch.Tensor, HDP_hierarchy: list):
    #     '''
    #     Check if the nCRP and HDP hierarchies match
    #     '''
    #     for idx, level_HDP in enumerate(HDP_hierarchy):
    #         level_nCRP = nCRP_hierarchy[:, :idx+1]
    #         level_nCRP_list = level_nCRP.tolist()
    #         level_nCRP_keys = [''.join(map(str, row)) for row in level_nCRP_list]
    #         for key in level_nCRP_keys:
    #             if key not in level_HDP.keys():
    #                 raise ValueError("The nCRP hierarchy {} does not match the HDP hierarchy {}".format(key, list(level_HDP.keys())))
    #     print("The nCRP hierarchy matches the HDP hierarchy")
        
    # def visualize_HDP(self, HDP_distributions: list, labels: torch.Tensor):
    #     '''
    #     Visualize the Hierarchical Dirichlet Process
    #     '''
    #     for l, distributions in enumerate(HDP_distributions):
    #         print("Layer: ", l)
    #         print("Number of subclass: ", len(distributions))
    #         for d in distributions:
    #             print("Values: ", d["values"], "Weights: ", d["weights"])
    
    # def calculate_likelihood(self, variable_indices, HDP_distributions: list, labels: torch.Tensor):
    #     '''
    #     Calculate the likelihood of the Hierarchical Dirichlet Process
    #     '''
    #     pass

    # def infer_HDP(self, HDP_distributions: list, labels: torch.Tensor, hdp_structure: list):
    #     '''
    #     Infer the Hierarchical Dirichlet Process layer by layer
    #     '''
        
    #     pass
    
    # def match_HDP(self, HDP_distributions: list, hierarchy_tree: dict):
    #     '''
    #     Match the Hierarchical Dirichlet Process
    #     '''
    #     pass

    # def infer_nCRP(self, labels: torch.Tensor, hierarchy_tree: dict):
    #     '''
    #     Infer the nested Chinese Restaurant Process
    #     '''
    #     augmented_tree = add_key_to_nested_dict(hierarchy_tree, -1, 0.5) # Exact key and alpha value remain to be determined
    #     category_indices = self.get_flat_index(augmented_tree, labels)
    #     category_counts = torch.tensor(jax.tree_util.tree_leaves(augmented_tree))
    #     augmented_structure = jax.tree.structure(augmented_tree)
    #     prior_params = self.generate_categorical_parameters(category_indices, category_counts)
    #     likelihood_params = self.get_likelihood_params(prior_params)
    #     posterior_params = self.calculate_posterior(prior_params, likelihood_params)

    #     new_categories = torch.distributions.Categorical(posterior_params).sample()
    #     new_augmented_tree = self.update_categories(new_categories, category_counts, augmented_structure)
    #     new_hierarchy_tree = modify_key_to_nested_dict(new_augmented_tree, -1, 0.5) # Exact key and alpha value remain to be determined
    #     return new_hierarchy_tree

    # def calculate_conditional_density(self, mixture_index: int, sample_index: tuple, labels: torch.Tensor, hierarchy_tree: dict):
    #     '''
    #     Calculate the conditional density of the nested Chinese Restaurant Process
    #     '''
    #     pass

    # def calculate_posterior(self, prior_params: torch.Tensor, likelihood_params: torch.Tensor):
    #     '''
    #     Calculate the posterior of the nested Chinese
    #     '''
    #     pass

    # def get_likelihood_params(self, prior_params: torch.Tensor):
    #     '''
    #     Get the likelihood parameters
    #     '''
    #     pass

    # def infer_DP_distributions(self, hierarchy_tree: dict, labels: torch.Tensor):
    #     '''
    #     Infer the Dirichlet Process distributions
    #     '''
    #     pass 

    # def generate_categorical_parameters(self, category_indices: torch.Tensor, category_counts: torch.Tensor):
    #     '''
    #     Generate the categorical parameters for the nested Chinese Restaurant Process
    #     '''
    #     params = torch.unsqueeze(category_counts, dim=0)
    #     params = params.repeat(category_indices.shape[0], 1)
    #     params[torch.arange(params.size(0)), category_indices] -= 1
    #     return params

    # def update_categories(self, category_indices: torch.Tensor, category_counts: torch.Tensor, structure: Any):
    #     '''
    #     Generate the categorical parameters for the nested Chinese Restaurant Process
    #     '''
    #     params = torch.unsqueeze(category_counts, dim=0)
    #     params = params.repeat(category_indices.shape[0], 1)
    #     params[torch.arange(params.size(0)), category_indices] += 1
    #     new_augmented_tree = jax.tree_util.tree_unflatten(structure, params)

    #     return new_augmented_tree

    # def get_flat_index(self, hierarchy_tree: dict, labels: torch.Tensor):
    #     '''
    #     Get the flat index of the labels
    #     '''
    #     distributions = torch.tensor(jax.tree_util.tree_leaves(hierarchy_tree))
    #     flat_categories = torch.cumsum(distributions, dim=0)
    #     sorted_indices = sort_by_columns_with_original_indices(labels)
    #     category_indices = find_indices_of_smallest_entries_bigger_than(flat_categories, sorted_indices)
        
    #     return category_indices


if __name__ == "__main__":
    # dp = DirichletProcess(1)
    # dp.sample(100)
    # print(dp.get_values())
    # print(dp.get_weights())

    hp = HierarchicalDirichletProcess(10, 3, 100, 10, {2: 10})
    hp.print_base_weights()
    hp.print_parameters()
    hp.print_hyperparameters()
    hp.print_distributions()  
    hp.print_labels()
    hp.print_number_of_subcategories()

    # labels = hp.generate_CRF(50, 1)
    # print("labels")
    # print(labels)
    # num_categories_per_layer, hierarchy_tree = hp.get_hierarchy_info(labels)
    # print("num_categories_per_layer")
    # print(num_categories_per_layer)
    # print("hierarchy_tree")
    # print(hierarchy_tree)
    # hierarchy_param_tree = hp.generate_parameters(100, hierarchy_tree, 100)
    # print("hierarchy_param_tree")
    # print(hierarchy_param_tree)
    # hdp, hdp_structure = hp.generate_HDP(100, hierarchy_tree, labels)
    # print("HDP")
    # print(hdp)
    # print(hp.visualize_HDP(hdp, labels))
    # print("HDP structure")
    # print(hdp_structure)
    # import tomotopy as tp
    # import matplotlib.pyplot as plt

    # mdl = tp.LDAModel(k=20)
    # for line in open('sample.txt'):
    #     mdl.add_doc(line.strip().split())

    # log_likelihoods = []
    # iterations = []
    # for i in range(0, 100, 1):
    #     mdl.train(10)
    #     print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    #     log_likelihoods.append(mdl.ll_per_word)
    #     iterations.append(i)
    # plt.plot(iterations[10:], log_likelihoods[10:])
    # plt.xlabel("Iteration")
    # plt.ylabel("Joint distribution")
    # plt.title("Joint Probability under Gibbs Iterations Average over 10")
    # # plt.show()
    # plt.savefig("HDP_Gibbs_Sampling_Ave10.png")
    

    # for k in range(mdl.k):
    #     print('Top 10 words of topic #{}'.format(k))
    #     print(mdl.get_topic_words(k, top_n=10))

    # mdl.summary()