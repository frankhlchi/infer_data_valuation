# Graph Inference Data Valuation Framework
![Inference_Data_Valuation_Poster_1](https://github.com/user-attachments/assets/4e598383-17fc-4754-846b-7e0a77039379)
This repository contains code of Shapley-Guided Utility Learning for Effective Graph Inference Data Valuation.

## Scripts Overview

1. **preprocess.py**
   - Preprocesses the graph dataset for GNN training and evaluation.
   - Input: Raw graph data
   - Output: Processed graph data ready for GNN consumption

2. **train_gnn.py**
   - Trains the GNN model on the preprocessed data.
   - Input: Processed graph data
   - Output: Trained GNN model

3. **valid_perm_sample.py**
   - Generates validation permutation samples for model evaluation.
   - Input: Trained GNN model, validation data
   - Output: Validation permutation samples

4. **test_perm_sample.py**
   - Generates test permutation samples for final model evaluation.
   - Input: Trained GNN model, test data
   - Output: Test permutation samples

5. **atc_confidence_estimation.py**
   - Estimates confidence using the Adaptive Test Confidence (ATC) method.
   - Input: Validation permutation samples
   - Output: ATC confidence estimates

6. **atc_ne_confidence_estimation.py**
   - Estimates confidence using the ATC with Negative Entropy (ATC-NE) method.
   - Input: Validation permutation samples
   - Output: ATC-NE confidence estimates


7. **training_statistics.py**
   - Computes and saves training set statistics for use in performance prediction.
   - Input: Trained GNN model, training data
   - Output: Training statistics


8. **doc_performance_prediction.py**
   - Predicts model performance using the Difference of Confidence (DOC) method.
   - Input: Training statistics, validation permutation samples
   - Output: DOC performance predictions


9. **shapley_regression_pred_lasso.py**
    - Performs Shapley regression prediction using LASSO regularization (our proposed method).
    - Input: Shapley values, performance metrics
    - Output: Regression model for performance prediction

10. **shapley_estimation_drop_node.py**
   - Estimates Shapley values for nodes by dropping them from the graph.

