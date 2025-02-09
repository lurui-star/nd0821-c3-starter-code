"""
Author: Rui Lu
Date: December, 2024
This script holds configuration of the test section
"""
import os
from pathlib import Path
from pipeline.data import import_data, process_data
from sklearn.model_selection import train_test_split
import pytest
import sys
import yaml

# Dynamically add the 'src' directory to sys.path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, src_dir)

@pytest.fixture(scope="session")
def config():
    """
    Fixture to load the configuration from a YAML file.

    Returns:
        dict: Loaded configuration from config.yaml
    """
    # Open and load the YAML config file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Return the loaded config to be used in tests
    return config

@pytest.fixture(scope="session")
def load_data(config):
    """
    Fixture to load and process data for testing.

    Returns:
        tuple: Processed features (X) and labels (y)
    """
    # Check if the data exists at the given path
    data_path = config["main"]["data"]["pth"]
    # Import the data from the specified path
    df = import_data(data_path)
    # Process the data: handle categorical features and label
    X, y = process_data(
        df,
        config["main"]["data"]["categorical_features"],
        config["main"]["data"]["label"]
    )
    return X, y


@pytest.fixture(scope="session")
def train_test_split_fixture(config):
    """
    Fixture to load, process, and split the data into training and validation sets.

    Returns:
        tuple: Training and validation features (X_train, X_val) and labels (y_train, y_val)
    """
    # Get the data path from the config
    data_path = config["main"]["data"]["pth"]
    # Import the data
    df = import_data(data_path)

    # Process the data: handle categorical features and label
    X, y = process_data(
        df,
        config["main"]["data"]["categorical_features"],
        config["main"]["data"]["label"],
    )

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["main"]["train_test_split"]["test_size"],
        random_state=config["main"]["train_test_split"]["random_state"],
    )

    # Return the split data
    return X_train, X_val, y_train, y_val
