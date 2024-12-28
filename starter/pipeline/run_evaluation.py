"""
Author: Rui Lu
Date: December, 2024
This script run evaluate_model 
"""

import logging
import pickle
import os
import pandas as pd
from pipeline.utils import plot_roc_curve
from pipeline.utils import plot_feature_importance
from pipeline.utils import save_model
from pipeline.evaluate_functions import compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def run_evaluate_model(y, Y_pred, Y_pred_prob, best_model, X, output_dir=".",set_name="Test Set"):
    """
    Evaluate model performance and save results (metrics, plots, and model) to specified directory.
    
    Parameters:
    - y_val: True validation labels
    - Y_test_pred: Predicted labels from the model
    - Y_test_pred_prob: Predicted probabilities from the model
    - best_model: Trained model (e.g., RandomForest, etc.)
    - X: Validation features
    - output_dir: Directory where results will be saved (default is 'output')
    """
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Print evaluation message
    print(f"This is the evaluation for {set_name}")

    # Compute evaluation metrics
    precision, recall, fbeta, fpr, tpr, roc_auc = compute_model_metrics(y, Y_pred, Y_pred_prob)

    # Print metrics
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Beta: {fbeta}")
    print(f"ROC AUC: {roc_auc}")
    
    # Save evaluation metrics to a CSV file
    logging.info("save evaluation metrics")
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F-Beta', 'ROC AUC'],
        'Score': [precision, recall, fbeta, roc_auc]
    })
    metrics_file = os.path.join(output_dir, 'model_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Plot and save the ROC Curve
    plot_roc_curve(fpr, tpr, roc_auc,output_dir)
    
    # Plot and save Feature Importance
    plot_feature_importance(best_model, X,output_dir,10)
    logging.info("save model")
    # Save the model as a .pkl file
    save_model(best_model.named_steps['model'], output_dir)
    