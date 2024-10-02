# Step 6: Run ATC-NE confidence estimation
##########################################
import os
import pickle
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.metrics import mean_absolute_error
import argparse
from preprocess_data import load_preprocessed_data, load_dataset
from tqdm import tqdm

def load_data(args):
    split_data = load_preprocessed_data(args.dataset, sampling_method=args.sampling_method, 
                                        data_seed=args.data_seed, setting=args.setting, pmlp=args.pmlp)
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(args.device)
    
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_data['train_indices']] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_data['val_indices']] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_data['test_indices']] = True
    data.edge_index = torch.tensor(split_data['val_edge_index'], device=args.device)

    return data, split_data

def compute_negative_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-8), axis=1)

def load_all_validation_results(args):
    output_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        output_dir += f"_seed{args.data_seed}"
    output_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    valid_probs = []
    valid_labels = []
    print(f"Loading all validation results from {output_dir}")
    
    for sample_file in os.listdir(output_dir):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(output_dir, sample_file)
            print(f"Processing {sample_file}")
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            for step_data in sample_data.values():
                valid_probs.append(step_data['val_probs'])
                valid_labels.append(step_data['val_labels'])
    
    return np.concatenate(valid_probs), np.concatenate(valid_labels)

def compute_threshold(valid_probs, valid_labels):
    valid_entropy = compute_negative_entropy(valid_probs)
    valid_acc = (valid_probs.argmax(axis=1) == valid_labels).mean()
    sorted_entropy = np.sort(valid_entropy)
    threshold_idx = int(len(sorted_entropy) * valid_acc)
    threshold = sorted_entropy[threshold_idx]
    print(f"Computed threshold: {threshold:.4f} (validation accuracy: {valid_acc:.4f})")
    return threshold

def compute_validation_atc_ne(args, threshold):
    print("Computing ATC-NE for validation set...")
    valid_res_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        valid_res_dir += f"_seed{args.data_seed}"
    valid_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    estimate_acc_values = []
    true_acc_values = []
    
    for sample_file in tqdm(os.listdir(valid_res_dir), desc="Processing validation samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(valid_res_dir, sample_file)
            
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            for step, step_data in sample_data.items():
                val_probs = step_data['val_probs']
                val_entropy = compute_negative_entropy(val_probs)
                estimate_acc = (val_entropy <= threshold).mean()
                true_acc_values.append(step_data['val_acc'])
                estimate_acc_values.append(estimate_acc)
                
                # Add ATC-NE prediction to step data
                step_data['atc_ne_pred_acc'] = estimate_acc
            
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
    
    print(f"Validation ATC-NE computation completed. Number of estimates: {len(estimate_acc_values)}")
    return true_acc_values, estimate_acc_values

def estimate_all_test_accuracy(args, threshold):
    print("Estimating all test accuracy...")
    test_res_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        test_res_dir += f"_seed{args.data_seed}"
    test_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    estimate_acc_values = []
    true_acc_values = []
    
    for sample_file in tqdm(os.listdir(test_res_dir), desc="Processing test samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(test_res_dir, sample_file)
            
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            for step, step_data in sample_data.items():
                test_probs = step_data['test_probs']
                test_entropy = compute_negative_entropy(test_probs)
                estimate_acc = (test_entropy <= threshold).mean()
                true_acc_values.append(step_data['test_acc'])
                estimate_acc_values.append(estimate_acc)
                
                step_data['atc_ne_pred_acc'] = estimate_acc
            
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
    
    print(f"Test ATC-NE computation completed. Number of estimates: {len(estimate_acc_values)}")
    return true_acc_values, estimate_acc_values

def evaluate_atc_ne(args):
    print(f"Evaluating ATC-NE for {args.dataset} dataset...")
    valid_probs, valid_labels = load_all_validation_results(args)
    threshold = compute_threshold(valid_probs, valid_labels)
    
    # Compute ATC-NE for validation set
    val_true_acc_values, val_estimate_acc_values = compute_validation_atc_ne(args, threshold)
    
    # Compute MAE for validation set
    val_mae = mean_absolute_error(val_true_acc_values, val_estimate_acc_values) * 100
    
    # Compute ATC-NE for test set
    test_true_acc_values, test_estimate_acc_values = estimate_all_test_accuracy(args, threshold)
    
    # Compute MAE for test set
    test_mae = mean_absolute_error(test_true_acc_values, test_estimate_acc_values) * 100
    
    print(f'Threshold: {threshold:.4f}')
    print(f'MAE of ATC-NE estimation on validation set: {val_mae:.2f}%')
    print(f'MAE of ATC-NE estimation on test set: {test_mae:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Evaluate ATC confidence estimation with negative entropy')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='Data seed for random sampling')
    parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP setting')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: SGC)')
    args = parser.parse_args()

    args.device = torch.device(args.device)

    print(f"Evaluating {args.dataset} dataset using ATC with negative entropy...")
    print(f"Settings: sampling_method={args.sampling_method}, setting={args.setting}, pmlp={args.pmlp}, base_model={args.base_model}")
    evaluate_atc_ne(args)

if __name__ == "__main__":
    main()
