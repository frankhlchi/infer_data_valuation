"""
train_gnn.py

Train, evaluate and save the best GNN model 
on the given dataset using preprocessed splits.
"""

import argparse
import os
import random
import numpy as np
import torch
import warnings
import yaml
from models import SGCNet, GCNNet
from preprocess_data import load_preprocessed_data, load_dataset
import json

warnings.filterwarnings("ignore", category=FutureWarning)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_model_for_tuning(base_model, num_features, num_classes, hidden_dim, seed=0):
    if base_model.lower() == 'sgc':
        return SGCNet(num_features, num_classes, hidden_dim=hidden_dim, seed=seed)
    elif base_model.lower() == 'gcn':
        return GCNNet(num_features, num_classes, hidden_dim=hidden_dim, num_layers=2, dropout=0.5, seed=seed)
    else:
        raise ValueError(f"Unsupported model: {base_model}")

def tune_hyperparameters(train_data, val_data, test_data, num_features, num_classes, config, seed, device, base_model):
    best_val_acc = 0
    best_config = {}

    for hidden_dim in config['hidden_dims']:
        for num_epochs in config['num_epochs_list']:
            for lr in config['lr_list']:
                for weight_decay in map(float, config['weight_decay_list']):
                    model = create_model_for_tuning(base_model, num_features, num_classes, hidden_dim, seed)
                    model = model.to(device)
                    
                    model.fit(train_data, num_epochs, lr, weight_decay, device)
                    
                    # Evaluate on validation set
                    _, val_acc = model.predict_valid(val_data, device)
                    
                    # Evaluate on test set
                    _, test_acc = model.predict(test_data, device)
                    
                    print(f"Config: hidden_dim={hidden_dim}, num_epochs={num_epochs}, lr={lr}, weight_decay={weight_decay}")
                    print(f"Validation Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_config = {
                            'hidden_dim': hidden_dim,
                            'num_epochs': num_epochs,
                            'lr': lr,
                            'weight_decay': weight_decay
                        }
                        print(f"New best config found! Validation Accuracy: {val_acc:.4f}")
                    
                    print("--------------------")

    print(f"Best config: {best_config}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return best_config

def save_model_and_config(model, config, dataset_name, sampling_method, setting, pmlp, base_model, data_seed=None):
    # Create model and config directories if they don't exist
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('config'):
        os.makedirs('config')

    file_prefix = f"{dataset_name}_{sampling_method}"
    if sampling_method == 'random' and data_seed is not None:
        file_prefix += f"_seed{data_seed}"
    file_prefix += f"_{setting}_{'pmlp' if pmlp else 'gnn'}_{base_model}"
    
    # Save model
    model_path = os.path.join('model', f"{file_prefix}_best_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save config as YAML
    config_path = os.path.join('config', f"{file_prefix}_best_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"Best model saved as {model_path}")
    print(f"Best config saved as {config_path}")

def main():
    parser = argparse.ArgumentParser(description="Train, evaluate and save the best GNN model on the given dataset.")
    parser.add_argument('--dataset', type=str, default='Cora', help='Name of the dataset (default: Cora)')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing (default: default)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for data splitting (default: None)')
    parser.add_argument('--setting', type=str, default='inductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP (Parameterless MLP) setting')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: sgc)')
    args = parser.parse_args()

    # Check if CUDA is available and set device accordingly
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Fix random seed for reproducibility
    fix_seed(args.seed)

    # Load the original dataset
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(device)

    # Load the preprocessed split data
    split_data = load_preprocessed_data(
        args.dataset, sampling_method=args.sampling_method, seed=args.seed, 
        data_seed=args.data_seed, setting=args.setting, pmlp=args.pmlp
    )

    # Update the original data with preprocessed splits
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_data['train_indices']] = True
    
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_data['val_indices']] = True
    
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_data['test_indices']] = True

    # Create separate data objects for train, val, and test
    train_data = data.clone()
    train_data.edge_index = torch.tensor(split_data['train_edge_index'], device=device)

    val_data = data.clone()
    val_data.edge_index = torch.tensor(split_data['val_edge_index'], device=device)

    test_data = data.clone()
    test_data.edge_index = torch.tensor(split_data['test_edge_index'], device=device)

    # Load the hyperparameter search configuration
    with open('./config/config.yaml', 'r') as f:
        search_config = yaml.safe_load(f)

    config_key = f"{args.dataset}_{args.sampling_method}_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model.lower()}_split"
    if config_key not in search_config:
        raise ValueError(f"Configuration for {config_key} not found in config.yaml")

    # Tune hyperparameters
    best_config = tune_hyperparameters(train_data, val_data, test_data, 
        data.num_features, dataset.num_classes, 
        search_config[config_key], args.seed, device, args.base_model
    )

    # Train the best model
    best_model = create_model_for_tuning(args.base_model, data.num_features, dataset.num_classes, 
                                         hidden_dim=best_config['hidden_dim'], seed=args.seed)

    best_model.fit(train_data, num_epochs=best_config['num_epochs'], 
                   lr=best_config['lr'], weight_decay=best_config['weight_decay'], device=device)

    # Evaluate on validation set
    _, val_acc = best_model.predict_valid(val_data, device=device)
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    # Evaluate on test set
    _, test_acc = best_model.predict(test_data, device=device)
    print(f"Final Test Accuracy: {test_acc:.4f}")

    # Save the best model and config
    save_model_and_config(best_model, best_config, args.dataset, args.sampling_method, args.setting, args.pmlp, args.base_model, args.data_seed)


    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()
