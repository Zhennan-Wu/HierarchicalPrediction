import torch
import pyro
import random
from umap import UMAP
import matplotlib.pyplot as plt 
import numpy as np

from pyro.distributions import Categorical
from typing import Any, Union, List, Tuple, Dict
   

class HDP_DIST_INFO:
    def __init__(self, number_of_subcategories: dict, labels_group_by_categories: dict, latent_distribution_indices: torch.Tensor) -> None:
        self.number_of_subcategories = number_of_subcategories
        self.labels_group_by_categories = labels_group_by_categories
        self.latent_distribution_indices = latent_distribution_indices
    
    def get_number_of_subcategories(self):
        return self.number_of_subcategories
    
    def get_labels_group_by_categories(self):
        return self.labels_group_by_categories
    
    def get_latent_distribution_indices(self):
        return self.latent_distribution_indices
    
class INFO:
    def __init__(self, count: int, label: torch.Tensor, param: torch.Tensor) -> None:
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


def print_tree(d, file:str=None, indent:int =0, opened:bool = False):
    # Loop through dictionary items
    if (file != None):
        if (not opened):
            opened = True
            with open(file, 'w') as f:
                print("Reconstructed HDP distribution tree:", file=f)
            for key, value in d.items():
                # Print the key with proper indentation
                print('    ' * indent + str(key), file=open(file, 'a'))
                
                # If value is another dictionary, recursively call the function
                if isinstance(value, dict):
                    print_tree(value, file, indent + 1, opened)
                # If the value is a tensor, print its content
                elif isinstance(value, torch.Tensor):
                    print('    ' * (indent + 1) + f'Tensor: {value}', file=open(file, 'a'))
                else:
                    # Print the value with additional indentation
                    print('    ' * (indent + 1) + str(value), file=open(file, 'a'))
        else:
            for key, value in d.items():
                # Print the key with proper indentation
                print('    ' * indent + str(key), file=open(file, 'a'))
                # If value is another dictionary, recursively call the function
                if isinstance(value, dict):
                    print_tree(value, file, indent + 1, opened)
                # If the value is a tensor, print its content
                elif isinstance(value, torch.Tensor):
                    print('    ' * (indent + 1) + f'Tensor: {value}', file=open(file, 'a'))
                else:
                    # Print the value with additional indentation
                    print('    ' * (indent + 1) + str(value), file=open(file, 'a'))            
    else:  
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


def visualize_rbm(rbm, hidden_loader, level, savefig):
    X_train_bin, y_train = hidden_loader.dataset.tensors
    X_train_bin = X_train_bin.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_train_input = y_train.reshape(X_train_bin.shape[0], -1) / 10.0 
    X_train_embedded = rbm.transform(X_train_bin, y_train_input)
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("UMAP RBM Embedding of MNIST Training Data of level {}".format(level))
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    # plt.show()
    filename = savefig + "level_" + str(level) + ".png"
    plt.savefig(filename)
    plt.close()


def visualize_data(hidden_loader, level, savefig):
    X_train_bin, y_train = hidden_loader.dataset.tensors
    X_train_embedded = X_train_bin.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_train_input = y_train.reshape(X_train_bin.shape[0], -1) / 10.0
    
    umap = UMAP()
    # Fit and transform the data
    X_train_umap = umap.fit_transform(X_train_embedded)
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1], c=y_train, cmap="Spectral", s=1, alpha=0.6)
    plt.colorbar(scatter, label="Digit Label")
    plt.title("UMAP RBM Encoding of MNIST Training Data of level {}".format(level))
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(savefig)
    plt.close()
    # plt.show()    


def project_points_to_simplex(points):
    """
    Projects each point in a multidimensional cube [0, 1]^n onto the n-dimensional simplex.

    Parameters:
        points (np.ndarray): A 2D numpy array where each row is a point in the hypercube [0, 1]^n.

    Returns:
        np.ndarray: A 2D numpy array with each row projected onto the n-dimensional simplex.
    """
    # Number of points and dimension
    num_points, dim = points.shape
    
    # Array to store the projected points
    projected_points = np.zeros_like(points)
    
    for i in range(num_points):
        point = points[i]
        
        # Step 1: Sort the point in descending order
        u = np.sort(point)[::-1]
        
        # Step 2: Find the largest k such that the projection condition holds
        cumulative_sum = np.cumsum(u)
        rho = np.where(u > (cumulative_sum - 1) / (np.arange(dim) + 1))[0][-1]
        
        # Step 3: Compute theta
        theta = (cumulative_sum[rho] - 1) / (rho + 1)
        
        # Step 4: Project point onto the simplex
        projected_points[i] = np.maximum(point - theta, 0)
    
    return projected_points