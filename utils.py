import torch
import copy


def find_indices_of_smallest_entries_bigger_than(sorted_tensor, elements):
    '''
    Find the indices of the smallest entries in the sorted_tensor that are bigger than the elements
    '''
    # Use torch.searchsorted for binary search
    indices = torch.searchsorted(sorted_tensor, elements, right=True)
    # Ensure indices that exceed the length of the sorted_tensor are set to None
    indices[indices == len(sorted_tensor)] = -1  # Use -1 to denote no such entry
    if (torch.any(indices == -1)):
        raise ValueError("Warning: Some elements exceed the largest element in the sorted_tensor")
    return indices


def sort_by_columns_with_original_indices(original_labels: torch.Tensor):
    '''
    Get the index of labels in the flattened hierarchical tree
    '''
    labels = copy.deepcopy(original_labels)
    indices = torch.arange(labels.size(0))  # Initialize the indices
    # Iterate over columns in reverse order
    for col in range(labels.size(1) - 1, -1, -1):
        sorted_indices = labels[:, col].argsort()
        labels = labels[sorted_indices]
        indices = indices[sorted_indices]
    # To get the positions of the original rows in the sorted tensor
    original_positions = torch.argsort(indices)
    return original_positions