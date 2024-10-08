import torch
import pyro
import random

from pyro.distributions import Categorical
from typing import Any, Union, List, Tuple, Dict
   

class INFO:
    def __init__(self, count, label, param) -> None:
        self.count = count
        self.label = label
        self.param = param
    
    def get_count(self):
        return self.count
    
    def get_label(self):
        return self.label
    
    def get_param(self):
        return self.param
    

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
        v_prime = torch.distributions.Beta(concentrate0, concentrate1).sample().item()
        v_values.append(v_prime)
        pi_final = v_prime
        for j in range(k):
            pi_final *= 1. - v_values[j]
        child_weights.append(pi_final)
    return torch.tensor(child_weights)


def transfer_index_tensor_to_tuple(index: torch.Tensor):
    '''
    Transfer the index tensor to string
    '''
    index_list = index.tolist()
    index_tuple = []
    for index in index_list:
        if (isinstance(index, int)):
            index_tuple.append((index,))
        else:
            index_tuple.append(tuple(index))
    return index_tuple


def transfer_index_tuple_to_tensor(indices: list):
    '''
    Transfer the index tuple to tensor
    '''
    index_tensor = []
    for index in indices:
        index_tensor.append(torch.tensor(index))
    return torch.stack(index_tensor)


def print_tree(d, indent=0):
    # Loop through dictionary items
    for key, value in d.items():
        # Print the key with proper indentation
        print('    ' * indent + str(key))
        
        # If value is another dictionary, recursively call the function
        if isinstance(value, dict):
            print_tree(value, indent + 1)
        # If the value is a tensor, print its content
        elif isinstance(value, torch.Tensor):
            print('    ' * (indent + 1) + f'Tensor: {value}')
        else:
            # Print the value with additional indentation
            print('    ' * (indent + 1) + str(value))
