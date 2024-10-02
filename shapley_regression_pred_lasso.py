import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

class LassoRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(LassoRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.intercept = torch.nn.Parameter(torch.zeros(1))
        self.apply_constraint()

    def forward(self, x):
        return self.linear(x).squeeze()  + self.intercept

    def apply_constraint(self):
        with torch.no_grad():
            self.linear.weight.data = torch.clamp(self.linear.weight.data, min=0)

def estimate_shapley_values(data_dir, methods):
    shapley_values = {method: {} for method in methods}
    
    interaction_methods = []
    for i, method1 in enumerate(methods[:-1]):  # Exclude 'true_acc'
        for method2 in methods[i:-1]:
            interaction_method = f"interaction({method1},{method2})"
            interaction_methods.append(interaction_method)
            shapley_values[interaction_method] = {}

    all_methods = methods + interaction_methods

    num_samples = len([f for f in os.listdir(data_dir) if f.startswith('sample_') and f.endswith('.pkl')])
    print(f"Number of samples: {num_samples}")

    for sample_file in tqdm(os.listdir(data_dir), desc="Processing samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(data_dir, sample_file)
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)

            prev_scores = {method: None for method in all_methods}

            for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
                for method in methods:
                    if method == 'true_acc':
                        current_score = step_data['val_acc']
                    elif method == 'negative_entropy':
                        current_score = step_data['negative_entropy']
                    elif method == 'propagated_max_probs':
                        current_score = np.mean(step_data['propagated_max_probs'])
                    elif method == 'propagated_target_probs':
                        current_score = np.mean(step_data['propagated_target_probs'])
                    elif method == 'cosine_similarity':
                        current_score = np.mean(step_data['cosine_similarities'])
                    elif method == 'confidence':
                        current_score = np.mean(step_data['predicted_class_confidences'])
                    elif method == 'argmax_confidence':
                        current_score = np.mean(step_data['max_probs'])
                    elif method in ['step', 'added_node_count', 'current_edge_count', 'edge_ratio', 'avg_edge_cosine_similarity', 'avg_max_class_cosine_similarity']:
                        current_score = step_data[method]
                    else:
                        current_score = step_data[method]
                
                    if prev_scores[method] is not None:
                        marginal_contribution = current_score - prev_scores[method]
                        new_node = step_data['node']
                        if new_node not in shapley_values[method]:
                            shapley_values[method][new_node] = 0
                        shapley_values[method][new_node] += marginal_contribution
                
                    prev_scores[method] = current_score

                for interaction_method in interaction_methods:
                    method1, method2 = interaction_method[12:-1].split(',')
                    if prev_scores[method1] is not None and prev_scores[method2] is not None:
                        current_score = prev_scores[method1] * prev_scores[method2]
                        
                        if prev_scores[interaction_method] is not None:
                            marginal_contribution = current_score - prev_scores[interaction_method]
                            new_node = step_data['node']
                            if new_node not in shapley_values[interaction_method]:
                                shapley_values[interaction_method][new_node] = 0
                            shapley_values[interaction_method][new_node] += marginal_contribution
                        
                        prev_scores[interaction_method] = current_score

    return shapley_values, interaction_methods

def prepare_regression_data(shapley_values, methods):
    X = []
    y = []
    
    nodes = shapley_values[methods[0]].keys()
    for node in nodes:
        features = [shapley_values[method][node] for method in methods[:-1]]  # Exclude 'true_acc' from features
        X.append(features)
        y.append(shapley_values['true_acc'][node])
    
    X = np.array(X)
    y = np.array(y)

    return X, y

def lasso_feature_selection(X, y, all_methods, device, k_folds=5):
    # Check for NaN values
    nan_features = np.isnan(X).any(axis=0)
    if nan_features.any():
        print("Features containing NaN values:")
        for i, has_nan in enumerate(nan_features):
            if has_nan:
                print(f"- {all_methods[i]}")
    
    X = np.nan_to_num(X, nan=0.0)
    input_dim = X.shape[1]
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    alphas = torch.logspace(-8, 2, 100, device=device)
    best_alpha = None
    best_model = None
    best_overall_score = float('-inf')

    for alpha in tqdm(alphas, desc="Searching for best alpha"):
        fold_scores = []
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).to(device)
            X_val_tensor = torch.FloatTensor(X_val).to(device)
            y_val_tensor = torch.FloatTensor(y_val).to(device)

            model = LassoRegression(input_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
            
            best_val_score = float('-inf')
            best_epoch_model = None
            patience = 500
            no_improve = 0
            
            for epoch in range(10000):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_train_tensor)
                loss = torch.nn.MSELoss()(y_pred, y_train_tensor) + alpha * model.linear.weight.abs().sum()
                loss.backward()
                optimizer.step()
                model.apply_constraint()

                model.eval()
                with torch.no_grad():
                    y_val_pred = model(X_val_tensor)
                    val_score = -torch.nn.MSELoss()(y_val_pred, y_val_tensor).item()
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch_model = model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= patience:
                    break

            fold_scores.append(best_val_score)
            
        avg_score = np.mean(fold_scores)
        if avg_score > best_overall_score:
            best_overall_score = avg_score
            best_alpha = alpha.item()
            best_model = best_epoch_model

    final_model = LassoRegression(input_dim).to(device)

    # Train final model on all data
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001, weight_decay=best_alpha)
    
    best_overall_score = float('-inf')
    best_epoch = 0
    no_improve = 0
    
    for epoch in range(10000):
        final_model.train()
        optimizer.zero_grad()
        y_pred = final_model(X_tensor)
        loss = torch.nn.MSELoss()(y_pred, y_tensor) + best_alpha * final_model.linear.weight.abs().sum()
        loss.backward()
        optimizer.step()
        final_model.apply_constraint()

        final_model.eval()
        with torch.no_grad():
            y_pred = final_model(X_tensor)
            score = -torch.nn.MSELoss()(y_pred, y_tensor).item()
        
        if score > best_overall_score:
            best_overall_score = score
            best_epoch = epoch
            best_model = final_model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 1000:  # Early stopping
            break

    final_model.load_state_dict(best_model)
    
    selected_features = torch.where(final_model.linear.weight.squeeze() > 1e-5)[0].cpu().numpy()
    
    print(f"\nBest alpha: {best_alpha:.10f}")
    print(f"Number of selected features: {len(selected_features)}")
    print("Selected features:")
    for i in selected_features:
        print(f"- {all_methods[i]}: {final_model.linear.weight.squeeze()[i].item():.6f}")
    
    return final_model, selected_features, best_alpha

def apply_regression_model(model, data_dir, all_methods, interaction_methods, is_validation=True, device='cpu'):
    true_acc_margins = []
    pred_acc_margins = []

    for sample_file in tqdm(os.listdir(data_dir), desc=f"Processing {'validation' if is_validation else 'test'} samples"):
        if sample_file.startswith('sample_') and sample_file.endswith('.pkl'):
            sample_path = os.path.join(data_dir, sample_file)
            with open(sample_path, 'rb') as f:
                sample_data = pickle.load(f)

            sample_true_accs = []
            sample_pred_accs = []

            for step, step_data in sorted(sample_data.items(), key=lambda x: int(x[0])):
                features = []
                for method in all_methods:
                    if method == 'true_acc':
                        continue
                    elif method == 'negative_entropy':
                        features.append(step_data['negative_entropy'])
                    elif method == 'propagated_max_probs':
                        features.append(np.mean(step_data['propagated_max_probs']))
                    elif method == 'propagated_target_probs':
                        features.append(np.mean(step_data['propagated_target_probs']))
                    elif method == 'cosine_similarity':
                        features.append(np.mean(step_data['cosine_similarities']))
                    elif method == 'confidence':
                        features.append(np.mean(step_data['predicted_class_confidences']))
                    elif method == 'argmax_confidence':
                        features.append(np.mean(step_data['max_probs']))
                    elif method in ['step', 'added_node_count', 'current_edge_count', 'edge_ratio', 'avg_edge_cosine_similarity', 'avg_max_class_cosine_similarity']:
                        features.append(step_data[method])
                    elif method in interaction_methods:
                        method1, method2 = method[12:-1].split(',')
                        features.append(features[all_methods.index(method1)] * features[all_methods.index(method2)])
                    else:
                        features.append(step_data[method])

                features = torch.FloatTensor(features).reshape(1, -1).to(device)
                with torch.no_grad():
                    pred_acc = model(features).item()

                step_data['score_pred_acc_lasso_int'] = pred_acc

                sample_true_accs.append(step_data['val_acc' if is_validation else 'test_acc'])
                sample_pred_accs.append(pred_acc)

            true_acc_margins.extend(np.diff(sample_true_accs))
            pred_acc_margins.extend(np.diff(sample_pred_accs))

            with open(sample_path, 'wb') as f:
                pickle.dump(sample_data, f)

    true_acc_margins = np.array(true_acc_margins)
    pred_acc_margins = np.array(pred_acc_margins)
    mask = np.isfinite(true_acc_margins) & np.isfinite(pred_acc_margins)
    true_acc_margins = true_acc_margins[mask]
    pred_acc_margins = pred_acc_margins[mask]

    if len(true_acc_margins) > 0 and len(pred_acc_margins) > 0:
        correlation, _ = stats.pearsonr(true_acc_margins, pred_acc_margins)
    else:
        print("Warning: No valid data points for correlation calculation")
        correlation = np.nan

    return correlation, true_acc_margins, pred_acc_margins

def save_selected_features(args, best_alpha, selected_features, all_methods, model):
    # Generate a descriptive title
    title = f"{args.dataset}_{args.sampling_method}_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}"

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the selected features and their weights to a file
    with open(os.path.join(args.output_dir, f"{title}_selected_features_shap.txt"), 'w') as f:
        f.write(f"Best alpha: {best_alpha:.10f}\n")
        f.write(f"Number of selected features: {len(selected_features)}\n")
        f.write("Selected features:\n")
        for i in selected_features:
            f.write(f"- {all_methods[i]}: {model.linear.weight.squeeze()[i].item():.6f}\n")


def main():
    parser = argparse.ArgumentParser(description='Estimate Shapley values, train regression model, and predict performance')
    parser.add_argument('--dataset', type=str, default='Cora', help='Dataset name')
    parser.add_argument('--sampling_method', type=str, default='default', help='Sampling method used in preprocessing')
    parser.add_argument('--setting', type=str, default='transductive', choices=['inductive', 'transductive'], help='Experiment setting')
    parser.add_argument('--pmlp', action='store_true', help='Use PMLP setting')
    parser.add_argument('--output_dir', type=str, default='./lasso_regression_results', help='Directory to save the outputs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
    parser.add_argument('--base_model', type=str, default='sgc', choices=['sgc', 'gcn'], help='Base model to use (default: SGC)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0, help='Data seed for random sampling')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    valid_res_dir = f'./valid_perm_results/{args.dataset}_{args.sampling_method}'
    test_res_dir = f'./test_perm_results/{args.dataset}_{args.sampling_method}'
    if args.sampling_method == 'random' and args.data_seed is not None:
        valid_res_dir += f"_seed{args.data_seed}"
        test_res_dir += f"_seed{args.data_seed}"
    valid_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"
    test_res_dir += f"_{args.setting}_{'pmlp' if args.pmlp else 'gnn'}_{args.base_model}_split"

    methods = [
        'cosine_similarity',
        'avg_edge_cosine_similarity', 
        'avg_max_class_cosine_similarity',
        'argmax_confidence',
        'confidence', 
        'negative_entropy',  
        'propagated_max_probs', 
        'propagated_target_probs',
        'max_conf_gap',
        'true_acc'
    ]

    print("Estimating Shapley values...")
    shapley_values, interaction_methods = estimate_shapley_values(valid_res_dir, methods)
    
    print("Preparing regression data...")
    all_methods = methods[:-1] + interaction_methods  # Exclude 'true_acc' and add interaction methods
    X, y = prepare_regression_data(shapley_values, all_methods + ['true_acc'])
    
    print("Performing Lasso feature selection...")
    lasso_model, selected_features,best_alpha = lasso_feature_selection(X, y, all_methods, device)

    print("Saving selected features...")
    save_selected_features(args, best_alpha, selected_features, all_methods, lasso_model)

    print("\nApplying Lasso regression model to validation data...")
    valid_correlation, valid_true_margins, valid_pred_margins = apply_regression_model(
        lasso_model, valid_res_dir, all_methods, interaction_methods, is_validation=True, device=device)
    print(f"Validation Margin Correlation: {valid_correlation:.4f}")
    
    print("\nApplying Lasso regression model to test data...")
    test_correlation, test_true_margins, test_pred_margins = apply_regression_model(
        lasso_model, test_res_dir, all_methods, interaction_methods, is_validation=False, device=device)
    print(f"Test Margin Correlation: {test_correlation:.4f}")

if __name__ == "__main__":
    main()
