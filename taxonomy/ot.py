import numpy as np

def create_hierarchy_matrix(taxonomy, index_dict):
    """
    Creates a hierarchy matrix H where H(i,j) = 1 if node i is on the path from root to node j, else 0.
    
    Args:
        taxonomy (dict): The nested taxonomy dictionary
        index_dict (dict): Dictionary mapping node names to their indices
    
    Returns:
        np.ndarray: The hierarchy matrix H
    """
    # Get the number of nodes
    n = len(index_dict)
    
    # Initialize the matrix H with zeros
    H = np.zeros((n, n), dtype=int)
    
    def get_path_to_node(target_key, taxonomy, current_path=None):
        """
        Finds the path from root to the target key in the taxonomy.
        
        Args:
            target_key (str): The node name to find the path to
            taxonomy (dict): The taxonomy dictionary
            current_path (list): List to store the path of node names
            
        Returns:
            list: List of node names in the path from root to target_key
        """
        if current_path is None:
            current_path = []
        
        for key, value in taxonomy.items():
            if key == target_key:
                return current_path + [key]
            if isinstance(value, dict):
                result = get_path_to_node(target_key, value, current_path + [key])
                if result:
                    return result
            elif isinstance(value, list) and target_key in value:
                return current_path + [key, target_key]
        return None
    
    # For each node, find its path from the root and set H(i,j) = 1 for nodes i in the path
    for target_key in index_dict:
        j = index_dict[target_key]
        path = get_path_to_node(target_key, taxonomy)
        if path:
            for path_key in path:
                if path_key in index_dict:
                    i = index_dict[path_key]
                    H[i, j] = 1
    
    return H


