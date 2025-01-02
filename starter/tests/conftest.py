"""
Author: Rui Lu
Date: December, 2024
This script holds confguration of the test section 
"""
import os
from pipeline.data import import_data, process_data
from sklearn.model_selection import train_test_split
import pytest
import yaml


@pytest.fixture(scope='session')
def config():
    """
    Fixture to load the configuration from a YAML file.

    Returns:
        dict: Loaded configuration from config.yaml
    """
    # Define the path to the folder where the config.yaml file is located
    config_folder = "/Users/ruilu/nd0821-c3-starter-code"

    # Construct the full path to the config.yaml file
    config_path = os.path.join(config_folder, "config.yaml")

    # Open and load the YAML config file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Return the loaded config to be used in tests
    return config


@pytest.fixture(scope='session')
def load_data(config):
    """
    Fixture to load and process data for testing.

    Returns:
        tuple: Processed features (X) and labels (y)
    """
    # Check if the data exists at the given path
    data_path = config['main']['data']['pth']
    if not os.path.exists(data_path):
        pytest.fail(f"Data not found at path: {data_path}")

    # Import the data from the specified path
    df = import_data(data_path)

    # Process the data: handle categorical features and label
    X, y = process_data(
        df, config["main"]["data"]["categorical_features"], config["main"]["data"]["label"])

    return X, y


@pytest.fixture(scope='session')
def train_test_split_fixture(config):
    """
    Fixture to load, process, and split the data into training and validation sets.

    Returns:
        tuple: Training and validation features (X_train, X_val) and labels (y_train, y_val)
    """
    # Get the data path from the config
    data_path = config['main']['data']['pth']

    # Check if the data exists at the given path
    if not os.path.exists(data_path):
        pytest.fail(f"Data not found at path: {data_path}")

    # Import the data
    df = import_data(data_path)

    # Process the data: handle categorical features and label
    X, y = process_data(
        df,
        config["main"]["data"]["categorical_features"],
        config["main"]["data"]["label"]
    )

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config["main"]["train_test_split"]["test_size"],
        random_state=config["main"]["train_test_split"]["random_state"]
    )

    # Return the split data
    return X_train, X_val, y_train, y_val
