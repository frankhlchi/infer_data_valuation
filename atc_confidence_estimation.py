
# Step 5: Run ATC confidence estimation
#######################################
import os
import pickle
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import argparse
from preprocess_data import load_preprocessed_data, load_dataset
from tqdm import tqdm

def load_all_validation_results(args):
    print("Loading all validation results...")
    output_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        output_dir += f"_seed{args.data_seed}"
    output_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    all_valid_probs = []
    all_valid_labels = []
    
    for sample_file in os.listdir(output_dir):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(output_dir, sample_file)
            print(f"Processing {sample_file}")
            
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            for step_data in sample_data.values():
                all_valid_probs.append(step_data['val_probs'])
                all_valid_labels.append(step_data['val_labels'])
    
    valid_probs_concat = np.concatenate(all_valid_probs)
    valid_labels_concat = np.concatenate(all_valid_labels)
    print(f"All validation results loaded. Shape of probs: {valid_probs_concat.shape}, Shape of labels: {valid_labels_concat.shape}")
    return valid_probs_concat, valid_labels_concat

def compute_threshold(valid_probs, valid_labels):
    print("Computing threshold...")
    valid_confidence = valid_probs.max(axis=1)
    valid_acc = (valid_probs.argmax(axis=1) == valid_labels).mean()
    sorted_confidence = np.sort(valid_confidence)
    threshold_idx = int(len(sorted_confidence) * (1 - valid_acc))
    threshold = sorted_confidence[threshold_idx]
    print(f"Validation accuracy: {valid_acc:.4f}")
    print(f"Computed threshold: {threshold:.4f}")
    return threshold

def estimate_all_test_accuracy(args, threshold):
    print("Estimating all test accuracy...")
    test_res_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        test_res_dir += f"_seed{args.data_seed}"
    test_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    all_estimate_acc_values = []
    all_true_acc_values = []
    
    for sample_file in os.listdir(test_res_dir):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(test_res_dir, sample_file)
            print(f"Processing {sample_file}")
            
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            for step, step_data in sample_data.items():
                test_probs = step_data['test_probs']
                test_confidence = test_probs.max(axis=1)
                estimate_acc = (test_confidence >= threshold).mean()
                all_true_acc_values.append(step_data['test_acc'])
                all_estimate_acc_values.append(estimate_acc)
                
                # Update the step data with ATC prediction
                step_data['atc_pred_acc'] = estimate_acc
            
            # Save the updated sample data
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
    
    print(f"All test accuracy estimation completed. Number of estimates: {len(all_estimate_acc_values)}")
    return all_true_acc_values, all_estimate_acc_values

def compute_validation_atc(args, threshold):
    print("Computing ATC for validation set...")
    valid_res_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        valid_res_dir += f"_seed{args.data_seed}"
    valid_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    all_estimate_acc_values = []
    all_true_acc_values = []
    
    for sample_file in tqdm(os.listdir(valid_res_dir), desc="Processing validation samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(valid_res_dir, sample_file)
            
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            for step, step_data in sample_data.items():
                val_probs = step_data['val_probs']
                val_confidence = val_probs.max(axis=1)
                estimate_acc = (val_confidence >= threshold).mean()
                all_true_acc_values.append(step_data['val_acc'])
                all_estimate_acc_values.append(estimate_acc)
                
                # Add ATC prediction to step data
                step_data['atc_pred_acc'] = estimate_acc
            
            # Save the updated sample data
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
    
    print(f"Validation ATC computation completed. Number of estimates: {len(all_estimate_acc_values)}")
    return all_true_acc_values, all_estimate_acc_values

def evaluate_atc(args):
    print(f"Evaluating ATC for {args.dataset} dataset...")
    # Load all validation results
    valid_probs, valid_labels = load_all_validation_results(args)
    
    # Compute threshold
    threshold = compute_threshold(valid_probs, valid_labels)
    
    # Compute ATC for validation set
    val_true_acc_values, val_estimate_acc_values = compute_validation_atc(args, threshold)
    
    # Compute MAE for validation set
    val_mae = mean_absolute_error(val_true_acc_values, val_estimate_acc_values) * 100
    
    # Estimate all test accuracy
    test_true_acc_values, test_estimate_acc_values = estimate_all_test_accuracy(args, threshold)
    
    # Compute MAE for test set
    test_mae = mean_absolute_error(test_true_acc_values, test_estimate_acc_values) * 100
    
    print(f'Threshold: {threshold:.4f}')
    print(f'MAE of ATC estimation on validation set: {val_mae:.2f}%')
    print(f'MAE of ATC estimation on test set: {test_mae:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Evaluate ATC confidence estimation')
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

    print(f"Starting evaluation for {args.dataset} dataset...")
    print(f"Settings: sampling_method={args.sampling_method}, setting={args.setting}, pmlp={args.pmlp}, base_model={args.base_model}")
    evaluate_atc(args)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
