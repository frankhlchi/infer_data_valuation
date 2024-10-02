import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy import stats
from models import create_model
from preprocess_data import load_preprocessed_data, load_dataset

def estimate_shapley_values(test_res_dir, methods, num_samples=30, num_steps=100):
    shapley_values = {method: {} for method in methods}
    
    for sample_file in tqdm(os.listdir(test_res_dir), desc="Processing samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(test_res_dir, sample_file)
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            prev_scores = {method: None for method in methods}
            
            for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
                for method in methods:
                    if method == 'score':
                        current_score = step_data['score_pred_acc']
                    elif method == 'true_acc':
                        current_score = step_data['test_acc']
                    elif method == 'atc':
                        current_score = step_data['atc_pred_acc']
                    elif method == 'atc_ne':
                        current_score = step_data['atc_ne_pred_acc']
                    elif method == 'doc':
                        current_score = step_data['doc_pred_acc']
                    elif method == 'enhance_shap':
                        current_score = step_data['pseudo_acc']
                    elif method == 'cosine_similarity':
                        current_score = np.mean(step_data['cosine_similarities'])
                    elif method == 'confidence':
                        current_score = np.mean(step_data['predicted_class_confidences'])
                    elif method == 'argmax_confidence':
                        current_score = np.mean(step_data['max_probs'])
                    elif method == 'soft_pseudo_acc':
                        current_score = step_data['soft_pseudo_test_acc']
                    elif method == 'hard_pseudo_acc':
                        current_score = step_data['hard_pseudo_test_acc']
                    elif method == 'shap_score':
                        current_score = step_data['score_pred_acc_shapley']
                    elif method == 'shap_score_int':
                        current_score = step_data['score_pred_acc_shapley_int']
                    elif method == 'shap_score_margin':
                        current_score = step_data['pred_acc_margin']
                    elif method == 'shap_lasso_rel':
                        current_score = step_data['shap_lasso_rel']
                    elif method == 'shap_lasso_sin':
                        current_score = step_data['shap_lasso_sin']
                    elif method == 'shap_lasso_enh':
                        current_score = step_data['shap_lasso_enh']
                    elif method == 'shap_score_margin_int':
                        current_score = step_data['pred_acc_margin_int']
                    elif method == 'score_pred_acc_lasso_int':
                        current_score = step_data['score_pred_acc_lasso_int']
                    elif method == 'point_pred_acc_lasso':
                        current_score = step_data['point_pred_acc_lasso']
                    elif method ==  'pred_acc_margin_lasso':
                        current_score = step_data['margin_pred_acc_lasso']
                    elif method == 'gnn_evaluator_pred_acc':
                        current_score = step_data['gnn_evaluator_pred_acc']
                    elif method == 'avg_edge_cosine_similarity':
                        current_score = step_data['avg_edge_cosine_similarity']
                    elif method == 'avg_min_class_cosine_similarity':
                        current_score = step_data['avg_min_class_cosine_similarity']
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    if prev_scores[method] is not None:
                        marginal_contribution = current_score - prev_scores[method]
                        new_node = step_data['node']
                        if new_node not in shapley_values[method]:
                            shapley_values[method][new_node] = 0
                        shapley_values[method][new_node] += marginal_contribution
                    
                    prev_scores[method] = current_score
    
    # Normalize Shapley values
    for method in methods:
        for node in shapley_values[method]:
            shapley_values[method][node] /= (num_samples * (num_steps - 1))
    
    return shapley_values

def main():
    parser = argparse.ArgumentParser(description='Estimate Shapley values and perform node removal')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--setting', type=str, default='transductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP setting')
    parser.add_argument('--output_dir', type=str, default='./shapley_results', help='Directory to save the outputs')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of samples for Shapley value estimation')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of steps for each sample')
    parser.add_argument('--device', type=str, default='cuda:3', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: SGC)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed for data splitting (default: 0)')
    args = parser.parse_args()

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print(f"Warning: CUDA is not available. Using CPU instead.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load preprocessed data
    split_data = load_preprocessed_data(args.dataset, sampling_method=args.sampling_method, setting=args.setting, pmlp=args.pmlp, data_seed=args.data_seed)
    dataset = load_dataset(args.dataset, './tmp/')
    data = dataset[0].to(device)

    # Load the trained model
    model_dir = "./model"
    model_name = f"{args.dataset}_{args.sampling_method}"
    if args.sampling_method == 'random' and args.data_seed is not None:
        model_name += f"_seed{args.data_seed}"
    model_name += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_best_model.pt"
    model_path = os.path.join(model_dir, model_name)
    
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
    model = model.to(device)
    
    # Estimate Shapley values
    test_res_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        test_res_dir += f"_seed{args.data_seed}"
    test_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    
    methods = ['gnn_evaluator_pred_acc', 'true_acc', 'atc', 'atc_ne', 'doc',  'confidence', 'argmax_confidence','shap_lasso_sin', 'point_pred_acc_lasso']
    #methods = ['score_pred_acc_lasso_int', 'point_pred_acc_lasso', 'pred_acc_margin_lasso']
    print(f"Estimating Shapley values for all methods...")
    shapley_values = estimate_shapley_values(test_res_dir, methods, num_samples=args.num_samples, num_steps=args.num_steps)

    for method in methods:
        print(f"Processing results for {method} method...")
        # Sort nodes by Shapley values
        sorted_nodes = sorted(shapley_values[method].items(), key=lambda x: x[1], reverse=True)
        sorted_test_nodes = [node for node, _ in sorted_nodes]

        # Perform node removal and evaluate accuracy
        accuracies = []
        data_clone = data.clone()
        data_clone.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data_clone.train_mask[split_data['train_indices']] = True
        data_clone.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data_clone.test_mask[split_data['test_indices']] = True
        data_clone.edge_index = torch.tensor(split_data['test_edge_index'], device=device)

        print(f"Performing node removal for {method} method...")

        # Evaluate initial accuracy without removing any nodes
        model.eval()
        with torch.no_grad():
            out, _ = model(data_clone)
            pred = out.argmax(dim=1)
            correct = float(pred[data_clone.test_mask].eq(data_clone.y[data_clone.test_mask]).sum().item())
            acc = correct / data_clone.test_mask.sum().item()
        accuracies.append(acc)
        
        for i in tqdm(range(1, len(sorted_test_nodes) + 1)):
            removed_nodes = set(sorted_test_nodes[:i])
            edge_mask = ~(torch.isin(data_clone.edge_index[0], torch.tensor(list(removed_nodes), device=device)) | 
                          torch.isin(data_clone.edge_index[1], torch.tensor(list(removed_nodes), device=device)))
            data_clone.edge_index = data_clone.edge_index[:, edge_mask]
            
            model.eval()
            with torch.no_grad():
                out, _ = model(data_clone)
                pred = out.argmax(dim=1)
                correct = float(pred[data_clone.test_mask].eq(data_clone.y[data_clone.test_mask]).sum().item())
                acc = correct / data_clone.test_mask.sum().item()
            accuracies.append(acc)

        # Save results
        output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.sampling_method}')
        if args.sampling_method == 'random' and args.data_seed is not None:
            output_dir += f"_seed{args.data_seed}"
        output_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f'{method}_shapley_values.pkl'), 'wb') as f:
            pickle.dump(shapley_values[method], f)
        with open(os.path.join(output_dir, f'{method}_sorted_test_nodes.pkl'), 'wb') as f:
            pickle.dump(sorted_test_nodes, f)
        with open(os.path.join(output_dir, f'{method}_accuracies.pkl'), 'wb') as f:
            pickle.dump(accuracies, f)

        print(f"Results for {method} method saved in {output_dir}")

if __name__ == "__main__":
    main()