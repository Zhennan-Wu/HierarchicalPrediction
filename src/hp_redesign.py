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
        # a list of dictionaries, to record the number of samples in each category
        self.hierarchical_observations = []
        # a list of dictionaries, to record label grouped by categories
        self.labels_group_by_categories = []
        # Initialize the base distribution
        beta = Gamma(1, 1).sample()
        self.hyperparameters["BASE"] = beta
        self.parameters = self.generate_parameters()
        # Initialize hierarchical structure
        eta = Gamma(1, 1).sample()
        self.hyperparameters["nCRP"] = eta
        self.labels = self.generate_nCRP()
        self.summarize_group_info()
        # Record hierarchical distributions
        beta = Gamma(1, 1).sample()
        self.hyperparameters["GLOBAL"] = beta
        self.base_weight = self.generate_base_weights()
        
        self.hierarchical_distributions = []
        self.hyperparameters["DP"] = {}
        self.generate_hierarchical_distributions()
        self.hierarchical_prior = copy.deepcopy(self.hierarchical_distributions)

        self.distribution_indices = torch.zeros(self.batch_size, dtype=torch.int)
        self.distributions = torch.zeros(self.batch_size, self.latent_dimension)
        self.smallest_category_distribution_on_labels = None
        self.get_distributions()

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

    def print_distributions(self):
        '''
        Print the distributions of the Hierarchical Dirichlet Process
        '''
        print(self.distributions)
    
    def print_distribution_indices(self):
        '''
        Print the distribution indices of the Hierarchical Dirichlet Process
        '''
        print(self.distribution_indices)

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
        return weights
    
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

        for l in range(self.layers):
            child_categories = torch.unique(label_hierarchy[:, :l+2], dim=0)
            parent_categories, number_of_children = torch.unique(child_categories[:, :-1], dim=0, return_counts=True)
            
            parent_keys = [transfer_index_tensor_to_string(pc) for pc in parent_categories]
            num_subcategories = dict(zip(parent_keys, number_of_children.tolist()))
            self.number_of_subcategories.append(num_subcategories)

            parent_categories, indices, number_of_observations = torch.unique(label_hierarchy[:, :l+1], dim=0, return_inverse = True, return_counts=True)
            parent_keys = [transfer_index_tensor_to_string(pc) for pc in parent_categories]

            num_observations = dict(zip(parent_keys, number_of_observations.tolist()))
            self.hierarchical_observations.append(num_observations)

            categorized_samples = [torch.where(indices == i)[0] for i in range(parent_categories.shape[0])]
            samples_group_by_categories = dict(zip(parent_keys, categorized_samples))
            self.labels_group_by_categories.append(samples_group_by_categories)

        parent_categories = torch.unique(label_hierarchy, dim=0)
        num_subcategories = {}
        for pc in parent_categories:
            index_string = transfer_index_tensor_to_string(pc)
            num_subcategories[index_string] = 1   
        self.number_of_subcategories.append(num_subcategories)
    
    def generate_hierarchical_distributions(self):
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
        self.hierarchical_distributions.append(dict(zip(child_categories, distributions)))

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
                parents_weights += [self.hierarchical_distributions[l][parent]]*nc
            
            with Pool(total_num_childs) as p:
                params = list(zip(etas, parents_weights, truncated_lengths))
                distributions = p.starmap(calc_sequential_stick_breaking_weight, params)
            self.hierarchical_distributions.append(dict(zip(children, distributions)))

    def get_distributions(self):
        '''
        Get the distribution of the Hierarchical Dirichlet Process
        '''
        category_indices = self.labels.tolist()
        category_labels = [''.join(map(str, cat)) for cat in category_indices]
        category_distribution_on_labels = [self.hierarchical_distributions[-1][cat] for cat in category_labels]
        self.smallest_category_distribution_on_labels = torch.tensor(category_distribution_on_labels)
        self.distribution_indices = Categorical(self.smallest_category_distribution_on_labels).sample()
        self.distributions = self.parameters[self.distribution_indices]

    def update_posteriors(self):
        '''
        Update the posteriors of the Hierarchical Dirichlet Process
        '''
        posteriors = []
        unique_values, counts = torch.unique(self.distribution_indices, return_counts=True)
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
            pass
        pass
        
    def update_prior(self):
        '''
        Update the prior of the Hierarchical Dirichlet Process
        '''
        pass

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

    def group_labels_by_category(self):
        '''
        Group the labels by category
        '''
        pass

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
        parameters = self.distribution_indices[indice]
        unique_parameters, count = torch.unique(parameters, return_counts=True)
        param_count = torch.zeros(self.truncate_length)
        param_count[unique_parameters.flatten()] += count
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
    # dp = DirichletProcess(1)
    # dp.sample(100)
    # print(dp.get_values())
    # print(dp.get_weights())

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
    hp.print_distribution_indices()
    print("Distribution parameters of each label")
    hp.print_distributions()
    print("Smallest category distribution on labels")
    hp.print_smallest_category_distribution_on_labels()
    print("Labels grouped by categories")
    hp.print_labels_group_by_categories()
    print("Hierarchical observations")
    hp.print_hierarchical_observations()


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