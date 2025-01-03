"""
Author: Rui Lu
Date: December 2024
This script contains a function to fetch data from a local directory and perform preliminary cleaning.
"""

import pandas as pd
import os
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def import_data(pth):
    """
    Imports data from the provided path and cleans the column names and categorical features.

    Inputs
    ------
    pth : str
        Path to the dataset file (CSV or other format supported by pandas).

    Returns
    -------
    X : pd.DataFrame
        Cleaned data with stripped column names and categorical features.
    """

    # Load the dataset from the given path
    logger.info("Downloading data set")
    file_path = os.path.abspath(os.path.join(os.getcwd(), pth))
    X = pd.read_csv(file_path)

    # Strip column names
    X.columns = X.columns.str.strip()

    return X


def process_data(X, categorical_features=[], label=None):
    """Process the data used in the machine learning pipeline.

    Processes the data by stripping spaces from categorical features and optionally
    separates out the label if provided. This function can be used for both training and
    inference/validation.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing the features and label.
    categorical_features : list of str, optional, default=[]
        List containing the names of the categorical features.
        Will strip spaces and ensure they are in string format.
    label : str or None, optional, default=None
        Name of the label column in X. If None, the function will return an empty DataFrame for y.

    Returns
    -------
    X_processed : pd.DataFrame
        Processed feature data (categorical features stripped of spaces).
    y : pd.Series or None
        Processed label data as a pandas Series, or None if label is None.
    """

    # Handle categorical features: strip spaces and convert to strings
    logger.info("Handle categorical features")
    if categorical_features:
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype(str).str.strip()

    # Separate label column from feature columns if label is provided
    logger.info("Encode response variable")
    if label is not None:
        if label not in X.columns:
            raise ValueError(f"Label column '{label}' not found in X.")
        y = X[label].map({"<=50K": 0, ">50K": 1})  # Return as pandas Series
        X = X.drop([label], axis=1)
    else:
        y = None  # Return None if no label is provided

    # Return processed features as a DataFrame
    X_processed = X.copy()  # Ensure we return a DataFrame
    return X_processed, y
