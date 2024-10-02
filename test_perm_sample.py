# Step 4: Run test permutation sampling
#######################################
import argparse
import os
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph
from preprocess_data import load_preprocessed_data, load_dataset
from models import create_model, SGConvNoWeight

def get_candidate_nodes(data, active_nodes, hop, exclude_nodes, device):
    candidate_nodes = set()
    for node in active_nodes:
        new_nodes, _, _, _ = k_hop_subgraph(node, hop, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)
        candidate_nodes.update(set(new_nodes.cpu().numpy().tolist()))
    candidate_nodes -= exclude_nodes
    return candidate_nodes

def get_2hop_neighbors(data, node):
    subset, _, _, _ = k_hop_subgraph(node, num_hops=2, edge_index=data.edge_index, 
                                     relabel_nodes=False, num_nodes=data.num_nodes)
    return set(subset.tolist())


def evaluate_subgraph(model, data, subgraph_nodes, device, train_prototype, test_predictions, neighbor_counts, step, full_edge_count, train_class_features, val_original_pred):
    # Create mask for subgraph nodes and edges
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[subgraph_nodes] = True
    edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
    subgraph_edge_index = data.edge_index[:, edge_mask]

    model.eval()
    conv = SGConvNoWeight(K=2).to(device)
    with torch.no_grad():
        # Use the subgraph for model inference
        subgraph_data = data.clone()
        subgraph_data.edge_index = subgraph_edge_index
        out, _ = model(subgraph_data) 
        probs = out.exp()
        _, pred = out.max(dim=1)
        test_probs = probs[data.test_mask]
        test_pred = pred[data.test_mask]
        test_labels = data.y[data.test_mask]
        test_acc = (test_pred == test_labels).float().mean().item()

        # Calculate negative entropy
        negative_entropy = -torch.sum(test_probs * torch.log(test_probs + 1e-8), dim=1).mean().item()

        # Compute additional information
        all_embeddings = conv(data.x, subgraph_edge_index)
        all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
        test_embeddings = all_embeddings[data.test_mask]

        cosine_similarities = F.cosine_similarity(test_embeddings, train_prototype.unsqueeze(0))
        max_probs_values, _ = test_probs.max(dim=1)
        max_probs = max_probs_values.cpu().numpy()

        # Get confidence for predicted classes
        predicted_class_confidences = test_probs[torch.arange(test_probs.size(0)), test_predictions].cpu().numpy()

        # Calculate new metrics
        current_edge_count = subgraph_edge_index.shape[1]
        edge_ratio = current_edge_count / full_edge_count

        # Calculate average cosine similarity of node features for added edges
        edge_cosine_similarities = F.cosine_similarity(
            data.x[subgraph_edge_index[0]],
            data.x[subgraph_edge_index[1]]
        )
        avg_edge_cosine_similarity = edge_cosine_similarities.mean().item()

        # Calculate class cosine similarities
        class_cosine_similarities = F.cosine_similarity(test_embeddings.unsqueeze(1), train_class_features.unsqueeze(0))
        max_class_cosine_similarities = class_cosine_similarities.max(dim=1)[0]
        avg_max_class_cosine_similarity = max_class_cosine_similarities.mean().item()

        # New calculations for propagating node feature predictions
        feature_data = data.clone()
        feature_data.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        feature_out, _ = model(feature_data)
        feature_probs = feature_out.exp()
        propagated_probs = conv(feature_probs, subgraph_edge_index)
        propagated_probs = F.normalize(propagated_probs, p=1, dim=1)  # Renormalize
        propagated_max_probs = propagated_probs.max(dim=1)[0][data.test_mask].cpu().numpy()
        propagated_target_probs = propagated_probs[data.test_mask, test_predictions].cpu().numpy()

        # New feature 1: max_conf_gap
        sorted_probs, _ = test_probs.sort(descending=True)
        max_conf_gap = (sorted_probs[:, 0] - sorted_probs[:, 1]).abs().mean().item()


    return {
        'test_probs': test_probs.cpu().numpy(),
        'test_pred': test_pred.cpu().numpy(),
        'test_labels': test_labels.cpu().numpy(),
        'test_acc': test_acc,
        'negative_entropy': negative_entropy,
        'cosine_similarities': cosine_similarities.cpu().numpy(),
        'max_probs': max_probs,
         'max_conf_gap': max_conf_gap,
        'predicted_class_confidences': predicted_class_confidences,
        'neighbor_counts': neighbor_counts,
        'step': step,
        'added_node_count': len(subgraph_nodes) - len(data.test_mask),
        'current_edge_count': current_edge_count,
        'edge_ratio': edge_ratio,
        'avg_edge_cosine_similarity': avg_edge_cosine_similarity,
        'avg_max_class_cosine_similarity': avg_max_class_cosine_similarity,
        'propagated_max_probs': propagated_max_probs,
        'propagated_target_probs': propagated_target_probs
    }


def sample_test_permutations(model, data, test_nodes, split_data, test_predictions, all_2hop_neighbors, val_original_pred, num_samples=30, random_seed=0, output_dir='./test_perm_results/', device='cpu'):
    conv = SGConvNoWeight(K=2).to(device)
    
    # Compute train prototype
    train_edge_index = split_data['train_edge_index']
    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long, device=device)
    
    # Create a subgraph for training data
    train_embeddings = conv(data.x, train_edge_index)[data.train_mask]
    train_prototype = train_embeddings.mean(dim=0)
    train_prototype = F.normalize(train_prototype, p=2, dim=0)

    # Compute train class features
    train_labels = data.y[data.train_mask]
    num_classes = train_labels.max().item() + 1

    train_class_features = []
    classes_present = []

    for c in range(num_classes):
        class_mask = train_labels == c
        class_count = class_mask.sum().item()
        
        if class_count > 0:
            class_embedding = train_embeddings[class_mask].mean(dim=0)
            train_class_features.append(class_embedding)
            classes_present.append(c)
        else:
            print(f"Warning: Class {c} not present in training set.")

    train_class_features = torch.stack(train_class_features)
    train_class_features = F.normalize(train_class_features, p=2, dim=1)

    # Count full test graph edges
    full_edge_count = split_data['test_edge_index'].shape[1]

    print('num_samples', num_samples)
    for i in range(num_samples):
        np.random.seed(random_seed + i)
        random.seed(random_seed + i)

        visited_nodes = set(test_nodes)
        all_2hop_neighbors_set = set.union(*[all_2hop_neighbors[node] for node in test_nodes])
        active_nodes = get_candidate_nodes(data, test_nodes, 1, visited_nodes, device)

        neighbor_counts = {node: 0 for node in test_nodes}

        # Initialize a dictionary to store all steps for this sample
        sample_results = {}

        # Evaluate initial graph (step 0)
        data_clone = data.clone()
        subgraph_nodes = list(visited_nodes)
        node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        node_mask[subgraph_nodes] = True
        edge_mask = node_mask[data_clone.edge_index[0]] & node_mask[data_clone.edge_index[1]]
        data_clone.edge_index = data_clone.edge_index[:, edge_mask]
        results = evaluate_subgraph(model, data_clone, list(visited_nodes), device, train_prototype, test_predictions, neighbor_counts, 0, full_edge_count, train_class_features, val_original_pred)
        results['node'] = None  # No new node added in step 0

        # Store step 0 results
        sample_results[0] = results

        step = 1
        while active_nodes and len(visited_nodes) < len(all_2hop_neighbors_set):
            new_node = random.sample(list(active_nodes), 1)[0]
            active_nodes.remove(new_node)
            visited_nodes.add(new_node)

            # Update neighbor counts
            for test_node in test_nodes:
                if new_node in all_2hop_neighbors[test_node]:
                    neighbor_counts[test_node] += 1

            # Add 1-hop neighbors of the new node to active_nodes
            new_candidates = get_candidate_nodes(data, {new_node}, 1, visited_nodes, device)
            active_nodes.update(new_candidates & all_2hop_neighbors_set)

            # Update data_clone.edge_index for the current subgraph
            data_clone = data.clone()
            subgraph_nodes = list(visited_nodes)
            node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            node_mask[subgraph_nodes] = True
            edge_mask = node_mask[data_clone.edge_index[0]] & node_mask[data_clone.edge_index[1]]
            data_clone.edge_index = data_clone.edge_index[:, edge_mask]

            # Evaluate subgraph
            results = evaluate_subgraph(model, data_clone, subgraph_nodes, device, train_prototype, test_predictions, neighbor_counts, step, full_edge_count, train_class_features, val_original_pred)
            results['node'] = new_node

            # Store step results
            sample_results[step] = results

            step += 1

        # Assert that all 2-hop neighbors have been visited
        assert visited_nodes == all_2hop_neighbors_set, "Not all 2-hop neighbors were visited"

        # Save all results for this sample in a single file
        sample_file = os.path.join(output_dir, f'sample_{i}.pkl')
        with open(sample_file, 'wb') as f:
            pickle.dump(sample_results, f)

        print(f"Sample {i + 1} completed. Total steps: {step - 1}")

def main():
    parser = argparse.ArgumentParser(description='Sample test permutations and evaluate model performance')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of permutation samples')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for sampling and prediction')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for data splitting')
    parser.add_argument('--setting', type=str, default='transductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP setting')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: SGC)')
    args = parser.parse_args()

    # Check if CUDA is available and set device accordingly
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load preprocessed data
    split_data = load_preprocessed_data(args.dataset, sampling_method=args.sampling_method, seed=args.seed, data_seed=args.data_seed, setting=args.setting, pmlp=args.pmlp)
    
    # Load the original dataset to get num_features and num_classes
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(device)

    # Create a PyG Data object from split_data
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_data['train_indices']] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_data['val_indices']] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_data['test_indices']] = True
    data.edge_index = torch.tensor(split_data['test_edge_index'], device=device)

    # Load the trained model
    model = create_model(args.base_model, 
                 num_features=data.num_features, 
                 num_classes=dataset.num_classes, 
                 dataset=args.dataset, 
                 sampling_method=args.sampling_method, 
                 setting=args.setting, 
                 pmlp=args.pmlp, 
                 seed=args.seed,
                 data_seed=args.data_seed).to(device)

    model_path = f'./model/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        model_path += f'_seed{args.data_seed}'
    model_path += f'_{args.setting}_{"pmlp" if args.pmlp else "gnn"}_{args.base_model}_best_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get test nodes
    test_nodes = split_data['test_indices']

    # Calculate 2-hop neighbors for all nodes
    all_2hop_neighbors = {node: get_2hop_neighbors(data, node) for node in range(data.num_nodes)}

    # Predict test labels using the full test graph
    with torch.no_grad():
        logits, _ = model(data)
        test_predictions = logits[data.test_mask].argmax(dim=1)

        data_val = data.clone()
        data_val.edge_index = torch.tensor(split_data['val_edge_index'], device=device)
        val_out, _ = model(data_val)
        val_original_pred = val_out[data.val_mask].argmax(dim=1)

    # Create output directory
    output_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        output_dir += f'_seed{args.data_seed}'
    output_dir += f'_{args.setting}_{"pmlp" if args.pmlp else "gnn"}_{args.base_model}_split'
    os.makedirs(output_dir, exist_ok=True)

    # Perform permutation sampling and evaluation
    sample_test_permutations(
        model, data, test_nodes, split_data, test_predictions, all_2hop_neighbors,
        val_original_pred, 
        num_samples=args.num_samples,
        random_seed=args.seed,
        output_dir=output_dir,
        device=device
    )

    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()
