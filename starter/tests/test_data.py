"""
Author: Rui Lu
Date: December, 2024
This script holds the test functions for dataset
"""

def test_columns_exist(load_data):
    """
    Test if all expected columns exist in the DataFrame.

    Args:
      load_data: Data to be tested (pandas DataFrame)
    """
    # Assume load_data is a DataFrame or similar structure
    X,y = load_data  # load_data is passed automatically by pytest as a fixture

    # List of expected columns
    expected_columns = [
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country'
    ]

    # Check if each expected column is in the DataFrame
    for column in expected_columns:
        assert column in X.columns, f"Column '{column}' is missing in the DataFrame."

    # If necessary, you can also print out the columns that were found for verification
    print("Found columns in DataFrame:", X.columns)



def test_column_dtypes(load_data):
    """
    Tests if columns are of correct datatypes

    Args:
        load_data: A fixture that loads the data (returns X, y)
    """
    # Define expected column datatypes
    expected_dtypes = {
        'age': 'int64',
        'workclass': 'object',
        'fnlgt': 'int64',
        'education': 'object',
        'education-num': 'int64',
        'marital-status': 'object',  
        'occupation': 'object',
        'relationship': 'object',
        'race': 'object',
        'sex': 'object',
        'capital-gain': 'int64',
        'capital-loss': 'int64',
        'hours-per-week': 'int64',  
        'native-country': 'object'
    }

    # Load the data (X is the features DataFrame)
    X, y = load_data  # `load_data` is passed as a fixture by pytest

    # Loop through expected column names and datatypes
    for column, expected_dtype in expected_dtypes.items():
        # Check if the column exists in the DataFrame
        assert column in X.columns, f"Column '{column}' is missing from the DataFrame."

        # Check if the datatype matches the expected type
        assert X[column].dtype.name == expected_dtype, f"Column '{column}' is not of type {expected_dtype}. Found {X[column].dtype.name} instead."

    print("Found columns and their types:", X.dtypes)