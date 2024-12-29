import os
from pipeline.data import import_data ,process_data
from sklearn.model_selection import train_test_split
import pytest
import yaml

# Define the path to the folder where the config.yaml file is located
config_folder = "/Users/ruilu/nd0821-c3-starter-code/starter/"
# Open the config.yaml file from the current directory
config_path = os.path.join(config_folder, "config.yaml")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)


@pytest.fixture(scope='session')
def load_data():
    """
    Data loaded from csv file used for tests

    Returns:
       X and y for tests
    """
    if not os.path.exists(config['main']['data']['pth']):
        pytest.fail(f"Data not found at path: {config['main']['data']['pth']}")
    df = import_data(config["main"]["data"]["pth"])

    X, y = process_data(df, config["main"]["data"]["categorical_features"], config["main"]["data"]["label"])
    return X,y


@pytest.fixture(scope='session')
def train_test_split():
    if not os.path.exists(config['main']['data']['pth']):
        pytest.fail(f"Data not found at path: {config['main']['data']['pth']}")
    
    df = import_data(config["main"]["data"]["pth"])
    X, y = process_data(df, config["main"]["data"]["categorical_features"], config["main"]["data"]["label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["main"]["train_test_split"]["test_size"], random_state=config["main"]["train_test_split"]["random_state"])

    return X_train, X_val, y_train, y_val