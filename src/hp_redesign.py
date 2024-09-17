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
    return torch.tensor(child_weights)


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
        ########################################################################################
        # fixed features
        ########################################################################################
        self.batch_size = batch_size 
        # int: the number of samples in each round
        
        self.truncate_length = truncated_length 
        # int: the total number of different parameters
        
        self.latent_dimension = latent_dimension 
        # int: the dimension of each sample
        
        self.layers = layers 
        # int: the number of layers in the Hierarchical Dirichlet Process
        
        self.layer_constrains = False 
        # bool: whether there are constraints on the number of categories in each layer
        
        self.implied_constraints = None 
        # dict: the implied constraints on the number of categories in each layer
        
        self.fixed_layers = fixed_layers 
        # dict: the constrains on the number of categories in each layer
        
        self._check_layer_constraints()
        # fill in the implied constraints from initial constraints input
        
        ########################################################################################
        # modifiable features from batch to batch
        ########################################################################################
        self.hyperparameters = {}
        # dict: the hyperparameters of the Hierarchical Dirichlet Process
        #  - BASE: the hyperparameter of the base dirichlet distribution
        #  - nCRP: the hyperparameter of the nested Chinese Restaurant Process
        #  - GLOBAL: the hyperparameter of the global distribution
        #  - DP: the hyperparameter of each level of Dirichlet Process

        self.number_of_subcategories = [] 
        # list of dict: the number of subcategories in each level under each category
        #  - key: the parent category
        #  - value: the number of subcategories

        self.hierarchical_observations = []
        # list of dict: the number of samples in each level under each category
        #  - key: the parent category
        #  - value: the number of samples

        self.labels_group_by_categories = []
        # list of dict: the samples belongs to each category in each level
        #  - key: the parent category
        #  - value: the samples belongs to the category

        self.hierarchical_distributions = []
        # list of dict: the hierarchical distributions of the Hierarchical Dirichlet Process
        #  - key: the parent category
        #  - value: the weights on all the available parameters (self.truncate_length is the total number of parameters)

        self.cumulative_weights = []
        # list of dict: the cumulative weights of all parameters in each category in each level across different batches
        #  - key: the parent category
        #  - value: the cumulative counts of all parameters

        self.hierarchical_prior = []
        # list of dict: the hierarchical distributions after all previous batches
        #  - key: the parent category
        #  - value: the weights on all the available parameters (self.truncate_length is the total number of parameters)

        self.latent_distribution_indices = torch.zeros(self.batch_size, dtype=torch.int)
        # torch.Tensor: the parameter indices of each sample

        self.distributions = torch.zeros(self.batch_size, self.latent_dimension)
        # torch.Tensor: the parameters of each sample

        self.smallest_category_distribution_on_labels = None
        # torch.Tensor: the basic distribution over parameters of all the labels

        self.hierarchy_tree = {}
        # Pytree: the categorical tree structure of the observations with leaf nodes as the the counts  and the parameter distribution under each category

        ################################################################################################
        # Generate all the available parameters
        ################################################################################################
        beta = Gamma(1, 1).sample()
        self.hyperparameters["BASE"] = beta
        self.parameters = self.generate_parameters()
        ################################################################################################
        # Initialize hierarchical structure
        ################################################################################################
        eta = Gamma(1, 1).sample()
        self.hyperparameters["nCRP"] = eta
        self.labels = self.generate_nCRP()
        self.number_of_subcategories, self.hierarchical_observations, self.labels_group_by_categories = self.summarize_group_info()
        ################################################################################################
        # Initialize the hierarchical distributions
        ################################################################################################
        beta = Gamma(1, 1).sample()
        self.hyperparameters["GLOBAL"] = beta
        self.base_weight = self.generate_base_weights()
        
        self.hyperparameters["DP"] = {}
        self.hierarchical_distributions = self.generate_hierarchical_distributions()

        self.hierarchical_prior = copy.deepcopy(self.hierarchical_distributions)

        self.smallest_category_distribution_on_labels, self.latent_distribution_indices, self.latent_distributions = self.summarize_distributions()

    def print_hyperparameters(self):
        '''
        Print the hyperparameters of the Hierarchical Dirichlet Process
        '''
        print(self.hyperparameters)
    
    def print_hierarchical_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        print(self.hierarchical_distributions)
    
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

    def print_latent_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        print(self.latent_distributions)
    
    def print_latent_distribution_indices(self):
        '''
        Print the distribution indices of the Hierarchical Dirichlet Process
        '''
        print(self.latent_distribution_indices)

    def print_smallest_category_distribution_on_labels(self):
        '''
        Print the smallest category distribution on the labels of the Hierarchical Dirichlet Process
        '''
        print(self.smallest_category_distribution_on_labels)

    def print_labels_group_by_categories(self):
        '''
        Print the labels grouped by categories
        '''
        print(self.labels_group_by_categories)
    
    def print_hierarchical_observations(self):
        '''
        Print the hierarchical observations of the Hierarchical Dirichlet Process
        '''
        print(self.hierarchical_observations)

    def print_cumulative_weights(self):
        '''
        Print the cumulative weights of the Hierarchical Dirichlet Process
        '''
        print(self.cumulative_weights)
    
    def print_prior(self):
        '''
        Print the prior of the Hierarchical Dirichlet Process
        '''
        print(self.hierarchical_prior)
    
    def print_hierarchy_tree(self):
        '''
        Print the observation tree of the Hierarchical Dirichlet Process
        '''
        print(self.hierarchy_tree)

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
           
        if (sum(weights) > 1 + 1e-3):
            raise ValueError("The sum of the weights should be smaller than 1, instead got {}".format(sum(weights)))
        return torch.tensor(weights)
    
    def generate_nCRP(self):
        '''
        Generate a nested Chinese Restaurant Process with sample size sample_size and concentration parameter eta

        Returns:
        - label_hierarchy (torch.Tensor): the labels of the samples in the nested Chinese Restaurant Process
        '''
        eta = self.hyperparameters["nCRP"]
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

    def summarize_group_info(self):
        '''
        Get the number of subcategories in the Hierarchical Dirichlet Process
        '''
        label_hierarchy = self.labels
        number_of_subcategories = []
        hierarchical_observations = []
        labels_group_by_categories = []

        for l in range(self.layers):
            if (l != self.layers - 1):
                child_categories = torch.unique(label_hierarchy[:, :l+2], dim=0)
                parent_categories, number_of_children = torch.unique(child_categories[:, :-1], dim=0, return_counts=True)
            
                parent_keys = [transfer_index_tensor_to_string(pc) for pc in parent_categories]
                one_layer_num_subcategories = dict(zip(parent_keys, number_of_children.tolist()))
                number_of_subcategories.append(one_layer_num_subcategories)   

            parent_categories, indices, number_of_observations = torch.unique(label_hierarchy[:, :l+1], dim=0, return_inverse = True, return_counts=True)
            parent_keys = [transfer_index_tensor_to_string(pc) for pc in parent_categories]

            num_observations = dict(zip(parent_keys, number_of_observations.tolist()))
            hierarchical_observations.append(num_observations)

            categorized_samples = [torch.where(indices == i)[0] for i in range(parent_categories.shape[0])]
            samples_group_by_categories = dict(zip(parent_keys, categorized_samples))
            labels_group_by_categories.append(samples_group_by_categories)

        parent_categories = torch.unique(label_hierarchy, dim=0)
        one_layer_num_subcategories = {}
        for pc in parent_categories:
            index_string = transfer_index_tensor_to_string(pc)
            one_layer_num_subcategories[index_string] = 1   
        number_of_subcategories.append(one_layer_num_subcategories)

        return number_of_subcategories, hierarchical_observations, labels_group_by_categories

    def generate_hierarchical_distributions(self):
        '''
        Generate the hierarchical tree from the label hierarchy
        '''
        hierarchical_distributions = []
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
        hierarchical_distributions.append(dict(zip(child_categories, distributions)))

        for l in range(self.layers-1):
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
                parents_weights += [hierarchical_distributions[l][parent]]*nc
            
            with Pool(total_num_childs) as p:
                params = list(zip(etas, parents_weights, truncated_lengths))
                distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
            hierarchical_distributions.append(dict(zip(children, distributions)))
        return hierarchical_distributions
    
    def summarize_distributions(self):
        '''
        Get the distribution of the Hierarchical Dirichlet Process
        '''
        category_indices = self.labels.tolist()
        category_labels = [''.join(map(str, cat)) for cat in category_indices]
        category_distribution_on_labels = [self.hierarchical_distributions[-1][cat] for cat in category_labels]
        
        smallest_category_distribution_on_labels = torch.stack(category_distribution_on_labels)
        latent_distribution_indices = Categorical(smallest_category_distribution_on_labels).sample()
        latent_distributions = self.parameters[latent_distribution_indices]

        return smallest_category_distribution_on_labels, latent_distribution_indices, latent_distributions

    def update_prior(self):
        '''
        Update the prior of the Hierarchical Dirichlet Process
        '''
        self.hierarchical_prior = copy.deepcopy(self.hierarchical_distributions)

    def update_cumulated_weights(self):
        '''
        Update the cumulated weights of the Hierarchical Dirichlet Process
        '''
        for level in range(self.layers):
            for category in self.number_of_subcategories[level].keys():
                indice = self.labels_group_by_categories[level][category]
                parameters = self.latent_distribution_indices[indice]
                unique_parameters, count = torch.unique(parameters, return_counts=True)
                param_count = torch.zeros(self.truncate_length)
                param_count[unique_parameters.flatten()] += count
                self.cumulative_weights[level][category]+= param_count   
        
    def posterior_update_of_distributions(self):
        '''
        Update the posteriors of the Hierarchical Dirichlet Process
        '''
        posteriors = []
        unique_values, counts = torch.unique(self.latent_distribution_indices, return_counts=True)
        evidence = torch.zeros(self.truncate_length)
        evidence[unique_values] += counts
        prior_param = self.hyperparameters["GLOBAL"].reshape(1,)
        evidence_param = torch.cat(prior_param, evidence)
        evidence_weights = Dirichlet(evidence_param).sample((self.truncate_length+1,))
        prior_weight = evidence_weights[0]
        likelihood_weight = evidence_weights[1:]
        posterior = {"0": prior_weight * self.hierarchical_prior[0]["BASE"] + likelihood_weight}
        self.hierarchical_distributions[0] = posterior
        for l in range(self.layers):
            with Pool(len(self.number_of_subcategories[l].keys())) as p:
                params = self._get_level_params_for_posterior(l)
                posteriors = p.starmap(calc_sequential_stick_breaking_weight, params)
            self.hierarchical_distributions[l+1] = dict(zip(self.hierarchical_distributions[l+1].keys(), posteriors))

    def posterior_update_of_labels(self):
        '''
        Update the labels of the Hierarchical Dirichlet Process
        '''
        observation_tree = self._generate_hierarchy_tree()
        augment_tree = self._augment_hierarchy_tree(observation_tree)
        v_counts, v_params, labels = self._separate_trees(augment_tree)
        str_labels = [''.join(map(str, label)) for label in self.labels]
        indices = []
        for idx, label in enumerate(str_labels):
            indices.append((idx, torch.where(labels == label)[0]))
        rows, cols = zip(*indices)
        v_counts[rows, cols] -= 1
        likelihood = v_params[torch.arange(self.batch_size), self.latent_distribution_indices, :]
        posterior = likelihood * v_counts
        new_label_indices = Categorical(posterior).sample()
        new_labels = labels[new_label_indices]
        label_ref = set(str_labels)
        new_label_ref = set(new_labels.tolist())
        if (not new_label_ref.issubset(label_ref)):
            # Generate new dsitributions and corresponding counts
            new_categories = list(new_label_ref - label_ref)
            self._increase_categories(new_categories)
        elif (not label_ref.issubset(new_label_ref)):
            # Remove redundant categories
            categories_to_remove = list(label_ref - new_label_ref)
            self._decrease_categories(categories_to_remove)

        self.labels = torch.tensor([[int(index) for index in label] for label in new_labels])
        self.number_of_subcategories, self.hierarchical_observations, self.labels_group_by_categories = self.summarize_group_info()

    def gibbs_update(self, number_of_iterations: int):
        '''
        Update the Hierarchical Dirichlet Process using Gibbs Sampling
        '''
        for _ in range(number_of_iterations):
            self.posterior_update_of_distributions()
            self.posterior_update_of_labels()
        
        self.update_cumulated_weights()
        self.update_prior()
        
    def _increase_categories(self, new_categories: list): 
        '''
        Increase the categories of the Hierarchical Dirichlet Process
        '''
        for new_cat in new_categories:
            for pi in range(len(new_cat)):
                parent_cat = new_cat[:pi+1]
                if (parent_cat not in self.self.hierarchical_distributions[pi].keys()):
                    self.hierarchical_distributions[pi][parent_cat] = calc_sequential_stick_breaking_weight(self.hyperparameters["DP"][parent_cat[:-1]], self.hierarchical_distributions[pi-1][parent_cat-1], self.truncate_length)

    def _decrease_categories(self, categories_to_remove: list):
        '''
        Decrease the categories of the Hierarchical Dirichlet Process
        '''
        for cat in categories_to_remove:
            for pi in range(len(cat)):
                parent_cat = cat[:pi+1]
                if (parent_cat in self.hierarchical_distributions[pi].keys()):
                    self.hierarchical_distributions[pi].pop(parent_cat)

    def _generate_hierarchy_tree(self):
        '''
        Generate the distribution tree from the hierarchical distributions
        '''
        root = {}
        tree_level = root

        for l in range(self.layers-1):
            if (l == 0):
                for cc in self.number_of_subcategories[l+1].keys():
                    pc = cc[:-1]
                    if (pc not in tree_level.keys()):
                        tree_level[pc] = {cc: {}}
                    else:
                        tree_level[pc][cc] = {}
                tree_level = tree_level.values()
            else:
                new_level = []
                for tree in tree_level:
                    for cc in self.number_of_subcategories[l+1].keys():
                        pc = cc[:-1]
                        if (pc in tree.keys()):
                            tree[pc][cc] = {}
                    new_level += tree.values()
                tree_level = new_level
        for tree in tree_level:
            for cc in self.number_of_subcategories[-1].keys():
                if (cc in tree.keys()):
                    tree[cc] = {'count': self.hierarchical_observations[-1][cc], "params": self.hierarchical_distributions[-1][cc], 'label': cc}
        return root

    def _augment_hierarchy_tree(self, observation_tree):
        '''
        Augment the observation tree with new potential categories at each level with leaves to be the hyperparameters
        '''
        tree_level = observation_tree
        for l in range(self.layers):
            if (l == 0):
                num_categories = len(tree_level.keys())
                tree_level[str(num_categories)] = {'parent': str(num_categories)}
                tree_level = tree_level.values()
            elif (l == self.layers - 1):
                for tree in tree_level:
                    if ('parent' in tree.keys()):
                        prefix = tree.pop('parent')
                    else:
                        prefix = list(tree.keys())[0][:-1]
                    num_categories = len(tree.keys())
                    new_value = prefix + str(num_categories)
                    tree[new_value] = {'count': self.hyperparameters["nCRP"].item(), 'params': self.base_weight}
            else:
                new_values = []
                for tree in tree_level:
                    if ('parent' in tree.keys()):
                        prefix = tree.pop('parent')
                    else:
                        prefix = list(tree.keys())[0][:-1]
                    num_categories = len(tree.keys())
                    new_value = prefix + str(num_categories)
                    tree[new_value] = {'parent': new_value}
                    new_values += tree.values()
                tree_level = new_values
        return observation_tree

    def _separate_trees(self, augment_tree):
        '''
        Separate the tree into subtrees
        '''
        augment_tree = augment_tree
        counts = []
        params = []
        labels = []
        flatten_leaves = jax.tree.leaves(augment_tree)
        for leaf in flatten_leaves:
            counts.append(leaf['count'])
            params.append(leaf['params'])
            labels.append(leaf['label'])
        counts = torch.tensor(counts)
        params = torch.stack(params).t()
        labels = torch.tensor(labels)
        vectorized_counts = torch.stack([counts]*self.batch_size)
        vectorized_params = torch.stack([params]*self.batch_size)
        return vectorized_counts, vectorized_params, labels

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

    def _get_level_params_for_posterior(self, level: int):
        '''
        Get the parameters for the posterior of the Hierarchical Dirichlet Process

        Parameters:
        - level (int): the level of the Hierarchical Dirichlet Process

        Returns:
        - prior_param (torch.Tensor): the prior parameters of the Hierarchical Dirichlet Process
        - evidence_param (torch.Tensor): the evidence parameters of the Hierarchical Dirichlet Process
        '''
        parent_categories = self.number_of_subcategories[level-1].keys()
        child_categories = self.number_of_subcategories[level].keys()
        parent_child_pairs = {}
        for pc in parent_categories:
            parent_child_pairs[pc] = []
            for cc in child_categories:
                if (pc == cc[:,-1]):
                    parent_child_pairs[pc].append(cc)
        params = []
        for child in child_categories:
            count = self._count_parameters_in_categories(child, level)
            prior_param = self.hyperparameters["DP"][child]
            prior = self.hierarchical_distributions[level-1][child[:-1]]
            params.append(tuple(count.sum().item(), (prior * prior_param + count)/(prior_param + count.sum())))
        return params

    def _count_parameters_in_categories(self, categories: str, level: int):
        '''
        Count the number of parameters in the categories

        Parameters:
        - categories (str): the categories to count the parameters
        - level (int): the level of the Hierarchical Dirichlet Process

        Returns:
        - num_parameters (int): the number of parameters in the categories
        '''
        indice = self.labels_group_by_categories[level][categories]
        parameters = self.latent_distribution_indices[indice]
        unique_parameters, count = torch.unique(parameters, return_counts=True)
        param_count = torch.zeros(self.truncate_length)
        param_count[unique_parameters.flatten()] += count
        param_count += self.cumulative_weights[level][categories]
        return param_count
        
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
    
    
if __name__ == "__main__":

    hp = HierarchicalDirichletProcess(10, 3, 100, 10, {2: 10})
    print("Base weights")
    hp.print_base_weights()
    print("Parameters pool")
    hp.print_parameters()
    print("HDP hyperparameters")
    hp.print_hyperparameters()
    print("Hierarchical distributions")
    hp.print_hierarchical_distributions()
    print("Labels")  
    hp.print_labels()
    print("Number of subcategories")
    hp.print_number_of_subcategories()
    print("Distribution indices of each label")
    hp.print_latent_distribution_indices()
    print("Distribution parameters of each label")
    hp.print_latent_distributions()
    print("Smallest category distribution on labels")
    hp.print_smallest_category_distribution_on_labels()
    print("Labels grouped by categories")
    hp.print_labels_group_by_categories()
    print("Hierarchical observations")
    hp.print_hierarchical_observations()
    print("Tree")
    hp.print_hierarchy_tree()
    print(hp.augment_hierarchy_tree())


    # labels = hp.generate_nCRP(50, 1)
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