import re
import pprint

def dict_to_pytree(tensor_dict):
    def add_to_pytree(key_path, tensor, tree=None):
        if len(key_path) == 0:
            return tensor
        
        i = key_path[0]
        rest = key_path[1:]
        if isinstance(i, int):
            if tree is None:
                tree = []

            # Extend the list
            tree = tree + [None] * (i + 1 - len(tree))
            tree[i] = add_to_pytree(rest, tensor, tree[i])
            return tree
        elif isinstance(i, str):
            if tree is None:
                tree = {}
            
            tree[i] = add_to_pytree(rest, tensor, tree.get(i, None))
            return tree
        else:
            raise ValueError("Unknown path part", i)

    def key_to_path(key):
        delimited_path = re.split("(#|\.)", key)
        parts = [delimited_path[0]]
        for delimeter, part in zip(delimited_path[1::2], delimited_path[2::2]):
            if delimeter == "#":
                parts.append(int(part))
            else:
                parts.append(part)
        return parts
    
    tree = {}
    for key, tensor in tensor_dict.items():
        path = key_to_path(key)
        tree = add_to_pytree(path, tensor, tree)

    return tree


def print_tree(pytree):
    def to_pretty_tree(pytree):
        if isinstance(pytree, dict):
            return {
                key: to_pretty_tree(value)
                for key, value in pytree.items()
            }
        elif isinstance(pytree, list):
            return [
                to_pretty_tree(value)
                for value in pytree
            ]
        else:
            return str(pytree.shape)
    
    pprint.pprint(to_pretty_tree(pytree))
