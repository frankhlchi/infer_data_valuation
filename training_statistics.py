# Step 7: Run DOC performance prediction (Training Statistics)
##############################################################
import argparse
import os
import pickle
import torch
import torch.nn.functional as F
from preprocess_data import load_preprocessed_data, load_dataset
from models import create_model

def compute_training_statistics(model, data, train_mask, device):
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        probs = F.softmax(logits[train_mask], dim=1)
        
        # Compute mean confidence
        mean_confidence = probs.mean().item()
        
        # Compute max confidence
        max_confidence = probs.max(dim=1)[0].mean().item()
        
        # Compute accuracy
        pred = logits[train_mask].argmax(dim=1)
        correct = (pred == data.y[train_mask]).sum().item()
        total = train_mask.sum().item()
        accuracy = correct / total
        
    return {
        'mean_confidence': mean_confidence,
        'max_confidence': max_confidence,
        'accuracy': accuracy
    }

def main():
    parser = argparse.ArgumentParser(description='Compute and save training set statistics')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='./train_statistics', help='Directory to save the outputs')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
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

    # Construct the model path
    model_dir = "./model"
    model_name = f"{args.dataset}_{args.sampling_method}"
    if args.sampling_method == 'random' and args.data_seed is not None:
        model_name += f"_seed{args.data_seed}"
    model_name += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_best_model.pt"
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load preprocessed data
    split_data = load_preprocessed_data(args.dataset, sampling_method=args.sampling_method, seed=args.seed, data_seed=args.data_seed, setting=args.setting, pmlp=args.pmlp)
    
    # Load the original dataset to get num_features and num_classes
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(device)

    # Create a PyG Data object from split_data
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_data['train_indices']] = True
    data.edge_index = torch.tensor(split_data['train_edge_index'], device=device)

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

    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Compute training statistics
    statistics = compute_training_statistics(model, data, data.train_mask, device)

    # Create output directory
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.sampling_method}')
    if args.sampling_method == 'random' and args.data_seed is not None:
        output_dir += f"_seed{args.data_seed}"
    output_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    os.makedirs(output_dir, exist_ok=True)

    # Save statistics to pkl file
    output_file = os.path.join(output_dir, 'train_statistics.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(statistics, f)

    print(f"Training statistics saved to {output_file}")

if __name__ == "__main__":
    main()
