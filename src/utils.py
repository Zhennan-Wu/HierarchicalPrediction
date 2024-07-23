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


def add_key_to_nested_dict(d, new_key, new_value):
    """
    Adds a new key with a given value to each level of a nested dictionary.

    Parameters:
    d (dict): The original nested dictionary.
    new_key (str): The key to add.
    new_value: The value to assign to the new key.

    Returns:
    dict: The modified dictionary with the new key added at each level.
    """
    if isinstance(d, dict):
        # Add the new key at the current level
        d[new_key] = new_value
        # Recursively add the new key to nested dictionaries
        for key in d:
            if isinstance(d[key], dict):
                add_key_to_nested_dict(d[key], new_key, new_value)
    return d


def modify_key_to_nested_dict(d, augmented_key, augmented_value):
    """
    Adds a new key with a given value to each level of a nested dictionary.

    Parameters:
    d (dict): The original nested dictionary.
    new_key (str): The key to add.
    new_value: The value to assign to the new key.

    Returns:
    dict: The modified dictionary with the new key added at each level.
    """
    if isinstance(d, dict):
        new_key = len(d)
        # Add the new key at the current level
        d[new_key] = d.pop(augmented_key) - augmented_value
        if (d[new_key] == 0):
            d.pop(new_key)
        # Recursively add the new key to nested dictionaries
        for key in d:
            if isinstance(d[key], dict):
                add_key_to_nested_dict(d[key], augmented_key, augmented_value)
    return d