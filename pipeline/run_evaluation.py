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
from pipeline.evaluate_functions import slice_compute_model_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run_evaluate_model(y, Y_pred, Y_pred_prob, best_model, X, categorical_features, output_dir=".", model_dir=".", set_name="Test Set",
                       slice_evaluation_by_feature=True):
    """
    Evaluates model performance and saves results, including metrics, plots, and the model itself, to the specified directory.

    Parameters:
    ----------
    y : array-like
        True labels or target values for the dataset (e.g., validation or test set).

    Y_pred : array-like
        Predicted labels (outputs from the model).

    Y_pred_prob : array-like
        Predicted probabilities from the model (for classification tasks).

    best_model : object
        The trained model (e.g., RandomForest, XGBoost) that was used to generate predictions.

    X : pd.DataFrame or np.ndarray
        Features used to train the model and to make predictions.

    output_dir : str, optional (default: ".")
        Directory where evaluation results (metrics, plots, model) will be saved.

    set_name : str, optional (default: "Test Set")
        A string representing the name of the dataset being evaluated (e.g., "Test Set" or "Validation Set").

    slice_evaluation_by_feature : bool, optional (default: True)
        If `True`, the model will be evaluated by slices of the categorical features (e.g., gender, age groups, etc.).

    categorical_features : list of str, optional (default: ['sex'])
        List of categorical features to evaluate the model slices by. This should be a list of column names in `X`.

    Returns:
    -------
    None
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Print evaluation message
    print(f"Evaluating model performance for {set_name}...")

    # Compute evaluation metrics
    meta_evaluation = compute_model_metrics(
        y, Y_pred, Y_pred_prob, slice_display=False)

    # Extract evaluation metrics
    precision = meta_evaluation['precision']
    recall = meta_evaluation['recall']
    fbeta = meta_evaluation['fbeta']
    fpr = meta_evaluation['fpr']
    tpr = meta_evaluation['tpr']
    roc_auc = meta_evaluation['roc_auc']

    # Print metrics to console
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F-Beta: {fbeta}")
    print(f"ROC AUC: {roc_auc}")

    # Save the evaluation metrics to a CSV file
    logging.info("Saving evaluation metrics to CSV...")
    metrics_df = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F-Beta', 'ROC AUC'],
        'Score': [precision, recall, fbeta, roc_auc]
    })
    metrics_file = os.path.join(output_dir, 'model_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")

    # Plot and save the ROC Curve
    logging.info("Plotting and saving ROC Curve...")
    plot_roc_curve(fpr, tpr, roc_auc, output_dir)

    # Plot and save Feature Importance
    logging.info("Plotting and saving Feature Importance...")
    plot_feature_importance(best_model, X, output_dir, max_features=10)

    # Save the model as a .pkl file
    logging.info("Saving the trained model...")
    save_model(best_model, model_dir)  # save model pipeline not only model

    # Evaluate the model by slices of categorical features (if applicable)
    if slice_evaluation_by_feature:
        logging.info("Evaluating by categorical features...")
        slice_compute_model_metrics(
            X, y, Y_pred, Y_pred_prob, categorical_features, output_dir)

    print(f"Model evaluation complete. Results saved to {output_dir}.")
