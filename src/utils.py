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


class TreeNode:
    def __init__(self, data, params):
        self.data = data  # Node's value
        self.params = params
        self.children = []  # List of child nodes
        self.parent = None  # Parent node (optional)
        self.num_leaves = 1  # Initially, the node is a leaf itself

    # Add a child node
    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child
        self.children.append(child_node)
        self.update_leaf_count_on_add()

    # Update leaf count when a child is added
    def update_leaf_count_on_add(self):
        if self.num_leaves == 1:  # If node was a leaf before
            self.num_leaves = 0  # Not a leaf anymore since it has children
        current_node = self
        while current_node:
            current_node.num_leaves = sum([child.num_leaves for child in current_node.children]) or 1
            current_node = current_node.parent

    # Remove a child node by data
    def remove_node(self, node_data):
        for child in self.children:
            if child.data == node_data:
                self.update_leaf_count_on_remove(child)
                self.children.remove(child)
                print(f"Node '{node_data}' removed successfully.")
                return True
        # If not found, recursively search in the subtrees
        for child in self.children:
            if child.remove_node(node_data):
                return True
        return False  # Return False if node not found

    # Update leaf count when a child is removed
    def update_leaf_count_on_remove(self, child_node):
        current_node = self
        while current_node:
            if len(current_node.children) == 1 and current_node.children[0] == child_node:
                current_node.num_leaves = 1  # Node becomes a leaf again if last child is removed
            else:
                current_node.num_leaves = sum([child.num_leaves for child in current_node.children]) or 1
            current_node = current_node.parent

    # Get the level of the node (for printing purposes)
    def get_level(self):
        level = 0
        current_node = self
        while current_node.parent:
            level += 1
            current_node = current_node.parent
        return level

    # Print the tree structure recursively along with the leaf count
    def print_tree(self):
        spaces = ' ' * self.get_level() * 4  # Indentation for levels
        prefix = spaces + "|-- " if self.parent else ""
        print(f"{prefix}{self.data} (leaves: {self.num_leaves})")
        for child in self.children:
            child.print_tree()

    # Pre-order traversal (root -> children)
    def pre_order(self):
        print(self.data, end=" ")  # Visit the root node
        for child in self.children:
            child.pre_order()  # Recursively visit each child

    # Post-order traversal (children -> root)
    def post_order(self):
        for child in self.children:
            child.post_order()  # Recursively visit each child
        print(self.data, end=" ")  # Visit the root node after visiting children

    # Level-order traversal (BFS - using a queue)
    def level_order(self):
        queue = [self]  # Start with the root node
        while queue:
            current_node = queue.pop(0)  # Remove the front node from the queue
            print(current_node.data, end=" ")  # Visit the current node
            queue.extend(current_node.children)  # Add all children to the queue
    
    def get_params(self):
        return self.params
    
    def update_params(self, new_params):
        self.params = new_params

# The Tree class that holds the root and provides tree-wide operations
class Tree:
    def __init__(self, root_data, root_params):
        self.root = TreeNode(root_data, root_params)  # Initialize the tree with a root node

    # Add a node to the tree by specifying parent node data
    def add_node(self, parent_data, child_data, child_params):
        parent_node = self.find_node(self.root, parent_data)
        if parent_node:
            new_node = TreeNode(child_data, child_params)
            parent_node.add_child(new_node)
            print(f"Added node '{child_data}' under parent '{parent_data}'.")
        else:
            print(f"Parent node '{parent_data}' not found.")

    # Remove a node by data
    def remove_node(self, node_data):
        if self.root.data == node_data:
            print("Cannot remove the root node.")
            return False
        if self.root.remove_node(node_data):
            return True
        else:
            print(f"Node '{node_data}' not found.")
            return False

    # Find a node in the tree by data (DFS search)
    def find_node(self, node, data):
        if node.data == data:
            return node
        for child in node.children:
            result = self.find_node(child, data)
            if result:
                return result
        return None

    # Print the entire tree structure
    def print_tree(self):
        if self.root:
            self.root.print_tree()
        else:
            print("The tree is empty.")

    # Perform pre-order traversal
    def pre_order(self):
        if self.root:
            self.root.pre_order()
            print()
        else:
            print("The tree is empty.")

    # Perform post-order traversal
    def post_order(self):
        if self.root:
            self.root.post_order()
            print()
        else:
            print("The tree is empty.")

    # Perform level-order traversal
    def level_order(self):
        if self.root:
            self.root.level_order()
            print()
        else:
            print("The tree is empty.")
