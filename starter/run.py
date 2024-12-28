"""
Author: Rui Lu
Date: December, 2024
This script used for run training, evaluting and saving the model
"""
import sys
import joblib
import logging
from sklearn.model_selection import train_test_split
from pipeline.data import import_data
from pipeline.data import process_data
from pipeline.run_evaluation import run_evaluate_model
from pipeline import utils
from pipeline.model import get_model_pipeline
from pipeline.model import train_model
from pipeline.model import train
from pipeline.model import inference
from pipeline.evaluate_functions import compute_model_metrics
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import argparse
import yaml

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def get_model_from_config(model_config: dict):
    """ Given a model configuration dict, return the corresponding model instance. """
    
    if 'XGBClassifier' in model_config:
        model_params = model_config['XGBClassifier']
        model = xgb.XGBClassifier(**model_params)  # Use parameters from config
        return model
    
    elif 'LogisticRegression' in model_config:
        model_params = model_config['LogisticRegression']
        model = LogisticRegression(**model_params)  # Use parameters from config
        return model
    
    else:
        raise ValueError("Unsupported model configuration.")
    

def go(config):
    logging.info("Import data from repo")
    df = import_data(config["main"]["data"]["pth"])
    logging.info("Process data to get features and target variable")
    X, y = process_data(df, config["main"]["data"]["categorical_features"], config["main"]["data"]["label"])
    logging.info("Split the data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["main"]["train_test_split"]["test_size"], random_state=config["main"]["train_test_split"]["random_state"])
    logging.info("Train the model with the specified parameters")
    best_model, best_params = train(get_model_from_config(config["main"]["modeling"]["MODEL"]), 
                                    X_train,
                                    y_train,
                                    config["main"]["modeling"]["param_grid"],
                                    config["main"]["modeling"]["FEATURES"])
    
    logging.info("Run inference on the validation set")
    Y_test_pred, Y_test_pred_prob = inference(best_model, X_val)
    logging.info("Evaluate the model performance")
    run_evaluate_model(y_val, Y_test_pred, Y_test_pred_prob, best_model, X_val, output_dir=config["main"]["modeling"]["output_dir"])

if __name__ == "__main__":
    go(config)