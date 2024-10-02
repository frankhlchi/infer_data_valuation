import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CoraFull, DBLP,WikiCS
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph, subgraph
import argparse
import os
import pickle
import random
import numpy as np
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import Actor, HeterophilousGraphDataset


def get_candidate_nodes(data, active_nodes, hop, exclude_nodes, device):
    """
    Get candidate nodes within 'hop' distance of 'active_nodes', excluding 'exclude_nodes'.
    """
    candidate_nodes = set()
    for node in active_nodes:
        new_nodes, _, _, _ = k_hop_subgraph(node, hop, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
        candidate_nodes.update(set(new_nodes.cpu().numpy().tolist()))
    candidate_nodes -= exclude_nodes
    return candidate_nodes

def load_dataset(dataset_name, root):
    """
    Load the specified dataset.
    """
    transform = T.NormalizeFeatures()
    
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=dataset_name, transform=transform)
    elif dataset_name == 'CoraFull':
        dataset = CoraFull(root=root, transform=transform)
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=root, name=dataset_name, transform=transform)
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=root, name=dataset_name, transform=transform)
    elif dataset_name == 'DBLP':
        dataset = DBLP(root=root, transform=transform)
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root=root, transform=transform)
    elif dataset_name in ['chameleon',  'squirrel']:
        dataset = WikipediaNetwork(root=root, name=dataset_name)
    elif dataset_name in [ 'crocodile']:
        dataset = WikipediaNetwork(root=root, name=dataset_name,geom_gcn_preprocess=False)
    elif dataset_name == 'actor':
        dataset = Actor(root=root)
    elif dataset_name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
        dataset = HeterophilousGraphDataset(root=root, name=dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset

def preprocess_data(dataset_name, root, device, cache_dir='./graph_split/', sampling_method='default', seed=0, data_seed=0, setting='inductive', pmlp=True):
    """
    Preprocess the dataset, create training, validation and test subgraphs, and save the results.
    """
    if data_seed is not None:
        print('data seed', data_seed)
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        torch.cuda.manual_seed(data_seed)
        torch.cuda.manual_seed_all(data_seed)

    # Load and normalize the dataset
    dataset = load_dataset(dataset_name, root)
    data = dataset[0].to(device)

    if setting == 'inductive':
        if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:  # Planetoid datasets
            if pmlp:
                split_data = process_inductive_planetoid_pmlp(data, device, dataset_name, data_seed)
            else:
                split_data = process_inductive_planetoid_gnn(data, device, dataset_name, data_seed)
        else:
            if pmlp:
                split_data = process_inductive_non_planetoid_pmlp(data, device, dataset_name, data_seed)
            else:
                split_data = process_inductive_non_planetoid_gnn(data, device, sampling_method, data_seed)
    else:  # transductive
        if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:  # Planetoid datasets
            if pmlp:
                split_data = process_transductive_planetoid_pmlp(data, device, sampling_method)
            else:
                split_data = process_transductive_planetoid_gnn(data, device, sampling_method)
        else:
            if pmlp:
                split_data = process_transductive_non_planetoid_pmlp(data, device, sampling_method, data_seed)
            else:
                split_data = process_transductive_non_planetoid_gnn(data, device, sampling_method, data_seed, dataset_name)
                
    # Add metadata to split_data
    split_data['metadata'] = {
        'dataset': dataset_name,
        'setting': setting,
        'pmlp': pmlp,
        'sampling_method': sampling_method,
        'seed': seed,
        'data_seed': data_seed
    }

    # Save preprocessed data
    save_preprocessed_data(split_data, dataset_name, cache_dir, sampling_method, seed, data_seed, setting, pmlp)

    return split_data


def process_inductive_planetoid_pmlp(data, device, dataset_name, data_seed):
    np.random.seed(data_seed)
    
    num_nodes = data.num_nodes
    
    val_ratio, test_ratio = 0.05, 0.05
    if  dataset_name == 'Cora' or dataset_name == 'Citeseer':
        val_ratio, test_ratio = 0.1, 0.1

    print(f"Processing {dataset_name} dataset with {num_nodes} nodes")

    # Use original train mask
    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    
    # Get remaining nodes
    remaining_nodes = set(range(num_nodes)) - set(train_indices.tolist())
    remaining_nodes = list(remaining_nodes)

    # Sample nodes for val and test sets from remaining nodes
    num_val = int(num_nodes * val_ratio)
    num_test = int(num_nodes * test_ratio)
    
    # Use NumPy to create random permutation
    shuffled_indices = np.random.permutation(len(remaining_nodes))
    shuffled_remaining = np.array(remaining_nodes)[shuffled_indices]
    
    val_indices = shuffled_remaining[:num_val]
    test_indices = shuffled_remaining[num_val:num_val+num_test]

    # Convert to PyTorch tensors if necessary
    val_indices = torch.tensor(val_indices, device=device)
    test_indices = torch.tensor(test_indices, device=device)

    print(f"Split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Update masks
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True
    
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True

    # Prepare training data (without edges for inductive setting)
    train_edge_index = torch.tensor([[],[]], dtype=torch.long).to(device)

    # Get remaining nodes and distribute them
    used_nodes = set(train_indices.tolist() + val_indices.tolist() + test_indices.tolist())
    remaining_nodes = list(set(range(num_nodes)) - used_nodes)

    # Shuffle the remaining nodes
    np.random.shuffle(remaining_nodes)

    val_extra_count = int(len(remaining_nodes) * (val_ratio / (val_ratio + test_ratio)))
    val_extra_nodes = remaining_nodes[:val_extra_count]
    test_extra_nodes = remaining_nodes[val_extra_count:]

    # Combine original and extra nodes
    val_all_nodes = val_indices.tolist() + val_extra_nodes
    test_all_nodes = test_indices.tolist() + test_extra_nodes

    # Generate validation graph (2-hop subgraph)
    val_2hop_nodes, val_2hop_edge_index, _, _ = k_hop_subgraph(
        node_idx=val_indices.tolist(),
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes
    )

    # Filter edges for validation set
    val_all_nodes_set = set(val_all_nodes)
    val_edge_mask = torch.tensor([src.item() in val_all_nodes_set and dst.item() in val_all_nodes_set 
                                  for src, dst in val_2hop_edge_index.t()], dtype=torch.bool)
    val_edge_index = val_2hop_edge_index[:, val_edge_mask]

    # Generate testing graph (2-hop subgraph)
    test_2hop_nodes, test_2hop_edge_index, _, _ = k_hop_subgraph(
        node_idx=test_indices.tolist(),
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes
    )

    # Filter edges for test set
    test_all_nodes_set = set(test_all_nodes)
    test_edge_mask = torch.tensor([src.item() in test_all_nodes_set and dst.item() in test_all_nodes_set 
                                   for src, dst in test_2hop_edge_index.t()], dtype=torch.bool)
    test_edge_index = test_2hop_edge_index[:, test_edge_mask]

    print(f"Validation subgraph - Nodes: {len(val_all_nodes)}, Edges: {val_edge_index.shape[1]}")
    print(f"Test subgraph - Nodes: {len(test_all_nodes)}, Edges: {test_edge_index.shape[1]}")

    split_data = {
        'train_indices': train_indices.tolist(),
        'train_edge_index': train_edge_index.cpu().numpy(),
        'val_indices': val_indices.tolist(),
        'val_edge_index': val_edge_index.cpu().numpy(),
        'test_indices': test_indices.tolist(),
        'test_edge_index': test_edge_index.cpu().numpy(),
        'val_all_nodes': val_all_nodes,
        'test_all_nodes': test_all_nodes,
    }

    print("Data split completed")
    return split_data


def process_transductive_planetoid_pmlp(data, device, sampling_method):
    if sampling_method != 'default':
        raise NotImplementedError("Only default sampling method is implemented for transductive PMLP.")
    
    split_data = {
        'train_indices': data.train_mask.nonzero(as_tuple=True)[0].tolist(),
        'train_edge_index': torch.empty((2, 0), dtype=torch.long).cpu().numpy(),  # Empty edge list for train
        'val_indices': data.val_mask.nonzero(as_tuple=True)[0].tolist(),
        'val_edge_index': data.edge_index.cpu().numpy(),
        'test_indices': data.test_mask.nonzero(as_tuple=True)[0].tolist(),
        'test_edge_index': data.edge_index.cpu().numpy(),
    }
    
    return split_data

def process_transductive_planetoid_gnn(data, device, sampling_method):
    if sampling_method != 'default':
        raise NotImplementedError("Only default sampling method is implemented for transductive GNN.")
    
    split_data = {
        'train_indices': data.train_mask.nonzero(as_tuple=True)[0].tolist(),
        'train_edge_index': data.edge_index.cpu().numpy(),
        'val_indices': data.val_mask.nonzero(as_tuple=True)[0].tolist(),
        'val_edge_index': data.edge_index.cpu().numpy(),
        'test_indices': data.test_mask.nonzero(as_tuple=True)[0].tolist(),
        'test_edge_index': data.edge_index.cpu().numpy(),
    }
    
    return split_data

# Remove the implementation of process_inductive_planetoid_gnn
def process_inductive_planetoid_gnn(data, device, sampling_method):
    raise NotImplementedError("Inductive Planetoid GNN processing is not implemented yet.")


def process_inductive_non_planetoid_pmlp(data, device, dataset_name, data_seed):
    """
    Process non-Planetoid datasets for inductive PMLP setting.
    """
   
    return process_non_planetoid_pmlp(data, device, dataset_name, data_seed)

def sample_nodes(data, train_ratio=0.01, val_ratio=0.03, test_ratio=0.03, data_seed=0):
    np.random.seed(data_seed)
    num_nodes = data.num_nodes
    shuffled_indices = np.random.permutation(num_nodes)

    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:train_size+val_size+test_size]

    return train_indices, val_indices, test_indices

def process_non_planetoid_pmlp(data, device, dataset_name, data_seed):
    np.random.seed(data_seed)
    """
    Process non-Planetoid datasets (CS, Physics, CoraFull, Computers, Photo, DBLP, WikiCS) for inductive PMLP setting.
    """
    num_nodes = data.num_nodes
    train_ratio, val_ratio, test_ratio = 0.01, 0.03, 0.03
    if dataset_name == 'Physics':
        train_ratio, val_ratio, test_ratio = 0.01, 0.01, 0.01
    if dataset_name == 'CoraFull':
        train_ratio, val_ratio, test_ratio = 0.1, 0.01, 0.01
    elif dataset_name in [ 'chameleon', 'crocodile', 'squirrel','actor', 'Roman-empire', 'Minesweeper', 'Tolokers', 'Questions',  'Amazon-ratings']:
        print(dataset_name)
        train_ratio, val_ratio, test_ratio = 0.05, 0.05, 0.05
    
    
    print(f"Processing {dataset_name} dataset with {num_nodes} nodes")

    # Sample nodes for train, val, and test sets
    train_indices, val_indices, test_indices = sample_nodes(data, train_ratio, val_ratio, test_ratio)

    print(f"Initial split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Update masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True

    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True

    # Prepare training data (without edges for inductive setting)
    train_edge_index = torch.tensor([[],[]], dtype=torch.long).to(device)

    # Get remaining nodes and distribute them
    all_nodes = set(range(num_nodes))
    used_nodes = set(train_indices.tolist() + val_indices.tolist() + test_indices.tolist())
    remaining_nodes = list(all_nodes - used_nodes)
    
    # Shuffle the remaining nodes
    np.random.shuffle(remaining_nodes)

    val_extra_count = int(len(remaining_nodes) * (val_ratio / (val_ratio + test_ratio)))
    val_extra_nodes = remaining_nodes[:val_extra_count]
    test_extra_nodes = remaining_nodes[val_extra_count:]

    # Combine original and extra nodes
    val_all_nodes = val_indices.tolist() + val_extra_nodes
    test_all_nodes = test_indices.tolist() + test_extra_nodes

    # Generate validation graph (2-hop subgraph)
    val_2hop_nodes, val_2hop_edge_index, _, _ = k_hop_subgraph(
        node_idx=val_indices.tolist(),
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes
    )

    # Filter edges for validation set
    val_all_nodes_set = set(val_all_nodes)
    val_edge_mask = torch.tensor([src.item() in val_all_nodes_set and dst.item() in val_all_nodes_set 
                                  for src, dst in val_2hop_edge_index.t()], dtype=torch.bool)
    val_edge_index = val_2hop_edge_index[:, val_edge_mask]

    # Generate testing graph (2-hop subgraph)
    test_2hop_nodes, test_2hop_edge_index, _, _ = k_hop_subgraph(
        node_idx=test_indices.tolist(),
        num_hops=2,
        edge_index=data.edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes
    )

    # Filter edges for test set
    test_all_nodes_set = set(test_all_nodes)
    test_edge_mask = torch.tensor([src.item() in test_all_nodes_set and dst.item() in test_all_nodes_set 
                                   for src, dst in test_2hop_edge_index.t()], dtype=torch.bool)
    test_edge_index = test_2hop_edge_index[:, test_edge_mask]

    print(f"Validation subgraph - Nodes: {len(val_all_nodes)}, Edges: {val_edge_index.shape[1]}")
    print(f"Test subgraph - Nodes: {len(test_all_nodes)}, Edges: {test_edge_index.shape[1]}")

    split_data = {
        'train_indices': train_indices.tolist(),
        'train_edge_index': train_edge_index.cpu().numpy(),
        'val_indices': val_indices.tolist(),
        'val_edge_index': val_edge_index.cpu().numpy(),
        'test_indices': test_indices.tolist(),
        'test_edge_index': test_edge_index.cpu().numpy(),
        'val_all_nodes': val_all_nodes,
        'test_all_nodes': test_all_nodes,
    }

    print("Data split completed")
    return split_data

def process_inductive_non_planetoid_gnn(data, device, sampling_method):
    # TODO: Implement this function
    raise NotImplementedError("Inductive non-Planetoid GNN processing not implemented yet.")


def process_transductive_non_planetoid_gnn(data, device, sampling_method, data_seed, dataset_name):
    np.random.seed(data_seed)
    """
    Process non-Planetoid datasets for transductive GNN setting.
    """
    num_nodes = data.num_nodes
    train_ratio = val_ratio = test_ratio = 0.01
    if dataset_name == 'Physics':
        train_ratio, val_ratio, test_ratio = 0.001, 0.001, 0.001

    # Randomly shuffle node indices
    shuffled_indices = torch.randperm(num_nodes)
    
    # Split indices into train, validation, and test sets
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    test_size = int(num_nodes * test_ratio)
    
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:train_size+val_size+test_size]

    print(f"Initial split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Update masks
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_indices] = True
    
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[val_indices] = True
    
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask[test_indices] = True

    # Use all edges for train, validation, and test in transductive setting
    all_edges = data.edge_index.cpu().numpy()

    # Error checking
    assert len(train_indices) > 0, "Training set is empty"
    assert len(val_indices) > 0, "Validation set is empty"
    assert len(test_indices) > 0, "Test set is empty"

    split_data = {
        'train_indices': train_indices.tolist(),
        'train_edge_index': all_edges,
        'val_indices': val_indices.tolist(),
        'val_edge_index': all_edges,
        'test_indices': test_indices.tolist(),
        'test_edge_index': all_edges,
    }

    return split_data


def save_preprocessed_data(split_data, dataset_name, cache_dir, sampling_method, seed, data_seed, setting, pmlp):
    """
    Save preprocessed data to a file if the file does not already exist.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_prefix = f'{dataset_name}_{sampling_method}'
    
    if sampling_method == 'random' and data_seed is not None:
        file_prefix += f'_seed{data_seed}'
    
    file_prefix += f'_{setting}_{"pmlp" if pmlp else "gnn"}'
    file_path = os.path.join(cache_dir, f'{file_prefix}_split.pkl')

    with open(file_path, 'wb') as f:
        pickle.dump(split_data, f)
    print(f'File {file_path} saved successfully.')

def load_preprocessed_data(dataset_name, cache_dir='./graph_split/', sampling_method='default', seed=None, data_seed=None, setting='inductive', pmlp=True):
    """
    Load preprocessed training, validation and test data.
    """
    file_prefix = f'{dataset_name}_{sampling_method}'
    
    if sampling_method == 'random' and data_seed is not None:
        file_prefix += f'_seed{data_seed}'
    
    file_prefix += f'_{setting}_{"pmlp" if pmlp else "gnn"}'

    file_path = os.path.join(cache_dir, f'{file_prefix}_split.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data file not found: {file_path}")

    with open(file_path, 'rb') as f:
        split_data = pickle.load(f)

    return split_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess data for graph neural network experiments')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--root', type=str, default='/data/frank/inference/prob_res/infer_data_valuation/tmp/', help='Root directory for datasets')
    parser.add_argument('--cache_dir', type=str, default='./graph_split/', help='Directory to save the preprocessed data')
    parser.add_argument('--sampling_method', type=str, default='default', choices=['default', 'random'], help='Method for sampling validation nodes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for data splitting (default: 0)')
    parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP (Parameterless MLP) setting')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess_data(args.dataset, args.root, device, args.cache_dir, args.sampling_method, args.seed, args.data_seed, args.setting, args.pmlp)
