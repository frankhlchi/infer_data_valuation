
# Step 8: Run DOC performance prediction
########################################
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import argparse
import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from models import create_model
from preprocess_data import load_preprocessed_data, load_dataset
from tqdm import tqdm

def load_train_statistics(dataset, sampling_method, setting, pmlp, base_model, data_seed=None):
    stats_dir = f'./train_statistics/{dataset}_{sampling_method}'
    if sampling_method == 'random' and data_seed is not None:
        stats_dir += f"_seed{data_seed}"
    stats_dir += f"_{setting}_{'pmlp' if pmlp else 'gnn'}_{base_model}_split"
    with open(os.path.join(stats_dir, 'train_statistics.pkl'), 'rb') as f:
        return pickle.load(f)

def compute_doc(base_confidence, target_confidence):
    return target_confidence - base_confidence

def compute_validation_doc(args, reg, base_confidence, base_accuracy):
    print("Computing DoC for validation set...")
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
                target_confidence = np.mean(step_data['max_probs'])
                doc = compute_doc(base_confidence, target_confidence)
                
                pred_acc_change = reg.predict([[doc]])[0]
                pred_acc = base_accuracy + pred_acc_change
                
                true_acc_values.append(step_data['val_acc'])
                estimate_acc_values.append(pred_acc)
                
                # Add DoC prediction to step data
                step_data['doc_pred_acc'] = pred_acc
            
            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)
    
    print(f"Validation DoC computation completed. Number of estimates: {len(estimate_acc_values)}")
    return true_acc_values, estimate_acc_values


def main():
    parser = argparse.ArgumentParser(description='Predict model performance using DoC method')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--data_seed', type=int, default=0, help='Data seed for random sampling')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--setting', type=str, default='transductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP setting')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: SGC)')
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    print("Loading train statistics...")
    train_stats = load_train_statistics(args.dataset, args.sampling_method, args.setting, args.pmlp, args.base_model, args.data_seed)
    base_confidence = train_stats['max_confidence']
    base_accuracy = train_stats['accuracy']
    print(f"Base confidence: {base_confidence:.4f}, Base accuracy: {base_accuracy:.4f}")

    print("Loading dataset and model...")
    split_data = load_preprocessed_data(
        args.dataset, 
        sampling_method=args.sampling_method, 
        setting=args.setting, 
        pmlp=args.pmlp, 
        data_seed=args.data_seed
    )
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(device)

    print("Preparing validation data for regression model...")
    valid_features = []
    valid_accs = []
    valid_res_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        valid_res_dir += f"_seed{args.data_seed}"
    valid_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"

    if not os.path.exists(valid_res_dir):
        raise FileNotFoundError(f"Validation results directory not found: {valid_res_dir}")

    for sample_file in tqdm(os.listdir(valid_res_dir), desc="Processing validation samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(valid_res_dir, sample_file)
            try:
                with open(sample_path, 'rb') as f:
                    sample_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {sample_file}: {e}")
                continue
            
            for step_data in sample_data.values():
                target_confidence = np.mean(step_data['max_probs'])
                doc = compute_doc(base_confidence, target_confidence)
                
                valid_features.append([doc])
                valid_accs.append(step_data['val_acc'] - base_accuracy)

    print("Training linear regression model...")
    reg = LinearRegression()
    reg.fit(valid_features, valid_accs)
    print(f"Regression coefficients: {reg.coef_}, intercept: {reg.intercept_}")

    print("Computing DoC for validation set...")
    val_true_acc_values, val_estimate_acc_values = compute_validation_doc(args, reg, base_confidence, base_accuracy)
    val_mae = mean_absolute_error(val_true_acc_values, val_estimate_acc_values) * 100
    print(f'MAE of DoC estimation on validation set: {val_mae:.2f}%')

    print("Predicting on test set...")
    test_res_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        test_res_dir += f"_seed{args.data_seed}"
    test_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"

    if not os.path.exists(test_res_dir):
        raise FileNotFoundError(f"Test results directory not found: {test_res_dir}")

    test_estimate_acc_values = []
    test_true_acc_values = []

    for sample_file in tqdm(os.listdir(test_res_dir), desc="Processing test samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(test_res_dir, sample_file)
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)

            for step, step_data in sample_data.items():
                target_confidence = np.mean(step_data['max_probs'])
                doc = compute_doc(base_confidence, target_confidence)

                pred_acc_change = reg.predict([[doc]])[0]
                pred_acc = base_accuracy + pred_acc_change

                # Store prediction results
                step_data['doc_pred_acc'] = pred_acc

                test_true_acc_values.append(step_data['test_acc'])
                test_estimate_acc_values.append(pred_acc)

            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)

    # Calculate MAE of accuracy estimation for test set
    test_mae = mean_absolute_error(test_true_acc_values, test_estimate_acc_values) * 100

    print(f'MAE of DoC estimation on validation set: {val_mae:.2f}%')
    print(f'MAE of DoC estimation on test set: {test_mae:.2f}%')

if __name__ == "__main__":
    main()
