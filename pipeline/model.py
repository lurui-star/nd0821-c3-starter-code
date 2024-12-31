"""
Author: Rui Lu
Date: December, 2024
This script holds the model functions needed to build, train and evaluate the model
"""
import pandas as pd 
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_model_pipeline(model, feats):
    """
    Creates model pipeline with feature preprocessing steps for
    encoding, scaling, and handling missing data.

    Args:
        model (sklearn model): sklearn model, either XGBoostClassifier or LogisticRegression.
        feats (dict): Dictionary of feature sets for each step of the pipeline 
                      (should contain keys 'drop', 'categorical', 'numeric').

    Returns:
        model_pipe (sklearn.pipeline.Pipeline): A pipeline containing preprocessing and the model.
    
    Raises:
        ValueError: If the model is neither XGBClassifier nor LogisticRegression.
        KeyError: If the required keys are not present in the feats dictionary.
    """
    # Check if the provided model is of a valid type
    if not isinstance(model, (LogisticRegression, XGBClassifier)):
        raise ValueError("Model should be either XGBClassifier or LogisticRegression.")
    
    # Check if required keys are in the feats dictionary
    required_keys = ['categorical', 'numeric']
    for key in required_keys:
        if key not in feats:
            raise KeyError(f"Missing required key: '{key}' in feats dictionary.")

    # Preprocessing for categorical features
    if isinstance(model, XGBClassifier):
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=1000)
    elif isinstance(model, LogisticRegression):
        encoder = OneHotEncoder(handle_unknown='ignore')

    # Categorical feature preprocessor (imputation + encoding)
    categ_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),  # Impute missing values with the most frequent value
        encoder
    )

    # Numerical feature preprocessor (scaling)
    numeric_preproc = StandardScaler()

    # Feature preprocessor (column transformer)
    feats_preproc = ColumnTransformer([
        ('categorical', categ_preproc, feats['categorical']),  # Process categorical features
        ('numerical', numeric_preproc, feats['numeric'])  # Scale numerical features
    ], remainder='passthrough')  # Pass other features without modification

    # Model pipeline (feature preprocessing + model)
    model_pipe = Pipeline([
        ('features_preprocessor', feats_preproc),
        ('model', model)
    ])

    return model_pipe


def train_model(model, X_train, y_train, param_grid, scoring='accuracy', cv=3):
    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=1)
    X_train=pd.DataFrame(X_train)
    y_train=pd.DataFrame(y_train)
    # Fit the model with grid search on training data
    _ =grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Optionally return best hyperparameters as well
    best_params = grid_search.best_params_

    return best_model, best_params

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)  # Predicted labels
    pred_prob= model.predict_proba(X)
    return pred, pred_prob 


def train(model, X_train, y_train, param_grid, feats):
    """
    Creates and trains a pipline for the given model

    Args:
        model (sklearn model): sklearn model
        X_train (pandas dataframe): Train features data
        y_train (pandas dataframe): Train labels data
        param_grid (dict): Parameters grid check config.py
        feats (dict): dict of features for each step of the pipeline check config.py

    Returns:
        model_pipe (sklearn pipeline/model): trained sklearn model or pipeline
    """
    logging.info("Creating model pipeline")
    model_pipe = get_model_pipeline(model, feats)  
    logging.info(f"Training {model.__class__.__name__} model")
    best_model, best_params= train_model(model_pipe, X_train, y_train,param_grid,scoring='accuracy',cv=5)

    return best_model, best_params