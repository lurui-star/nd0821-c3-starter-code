"""
Author: Rui Lu
Date: December, 2024
This script holds the test functions for train and evaluation of the model
"""

import os
import logging
import pytest

# Model training and evaluation
from pipeline.model import train
from pipeline.run_evaluation import run_evaluate_model
from pipeline.model import inference
from pipeline.utils import check_file_exists

# XGBoost
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_model_from_config(model_config: dict):
    """Given a model configuration dict, return the corresponding model instance."""

    if "XGBClassifier" in model_config:
        model_params = model_config["XGBClassifier"]
        model = XGBClassifier(**model_params)  # Use parameters from config
        return model

    elif "LogisticRegression" in model_config:
        model_params = model_config["LogisticRegression"]
        # Use parameters from config
        model = LogisticRegression(**model_params)
        return model

    else:
        raise ValueError("Unsupported model configuration.")


def test_train_and_evaluate_model(train_test_split_fixture, config):
    try:
        # Get model from config and train
        X_train, X_val, y_train, y_val = train_test_split_fixture
        model = get_model_from_config(config["main"]["modeling"]["MODEL"])
        best_model, best_params = train(
            model,
            X_train,  # X_train
            y_train,  # y_train
            config["main"]["modeling"]["param_grid"],
            config["main"]["modeling"]["FEATURES"],
        )

        # Run model evaluation
        Y_test_pred, Y_test_pred_prob = inference(best_model, X_val)
        run_evaluate_model(
            y_val,
            Y_test_pred,
            Y_test_pred_prob,
            best_model,
            X_val,
            output_dir=os.path.abspath(os.path.join(
                os.getcwd(), config["main"]["modeling"]["output_dir"])),
            model_dir=os.path.abspath(os.path.join(
                os.getcwd(), config["main"]["modeling"]["model_dir"])),
            slice_evaluation_by_feature=config["main"]["modeling"]["slice_output"][
                "slice_evaluation_by_feature"
            ],
            categorical_features=config["main"]["modeling"]["slice_output"][
                "categorical_features"
            ],
        )

        logging.info("SUCCESS: Training and evaluating model")

    except Exception as e:
        logging.error(f"ERROR: Model training/evaluating failed. {str(e)}")
        pytest.fail(f"Test failed due to: {str(e)}")

    # Check if the model was saved
    model_name = "best_model.pkl"
    model_path = config["main"]["modeling"]["model_dir"]
    check_file_exists(model_path, model_name)

    # Check if ROC image exists
    roc_image_name = "roc_curve.png"
    image_path = config["main"]["modeling"]["output_dir"]
    check_file_exists(image_path, roc_image_name)

    # Check if Feature Importance image exists
    feature_importance_image_name = "feature_importance.png"
    check_file_exists(image_path, feature_importance_image_name)

    # Check if model evaluation file exists
    model_eval_file_name = "slice_output.txt"
    check_file_exists(image_path, model_eval_file_name)

    # Check if overall evaluation file exists
    overall_eval_file_name = "model_metrics.csv"
    check_file_exists(image_path, overall_eval_file_name)
