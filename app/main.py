import os
import yaml
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List
from enum import Enum

# Import custom modules for processing data and inference
from pipeline.data import process_data
from pipeline.model import inference
from app.schemas import Person, FeatureInfo

# Define the application
app = FastAPI(
    title="Udacity - Project 3",
    description="Deploying a Machine Learning Model on Heroku with FastAPI",
    version="0.1",
)

# Model path setup
MODEL_PATH = os.path.join("/Users/ruilu/nd0821-c3-starter-code/starter/model", "best_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature and Example Information
EXAMPLES_PATH = "/Users/ruilu/nd0821-c3-starter-code/starter/app/examples.yaml"
with open(EXAMPLES_PATH) as fp:
    examples = yaml.safe_load(fp)

# List of categorical features (ensure correct spelling)
categorical_features = ['marital-status', 'occupation', 'relationship', 'education', 'race', 'sex', 'workclass', 'native-country', 'salary']

label = 'salary'

# Greeting endpoint
@app.get("/")
async def greetings():
    return {"greeting": "Welcome to salary prediction!"}

# Function to extract example data
def example_data_extract(example_data):
    """
    Extract and return a DataFrame from the provided example data.
    The data will include different example types such as "<=50k" and ">50k".
    """
    # Define the example types you want to iterate over
    example_types = [
        "Class <=50k (Label 0)",
        "Class >50k (Label 1)",
        "Missing sample",
        "Error sample"
    ]

    # Initialize an empty list to collect DataFrames
    dataframes = []

    # Iterate over each example type
    for item in example_types:
        try:
            # Extract the data using the example type
            data = example_data['post_examples'].get(item, {}).get('value', None)

            if data is None:
                print(f"Warning: No data found for {item}. Skipping.")
                continue

            # Convert the data into a DataFrame
            df = pd.DataFrame([data])

            # Add a new column 'salary' with value 'NA'
            df['salary'] = "NA"

            # Optionally add the 'example_type' to keep track of the source
            df['example_type'] = item

            # Append the current DataFrame to the list
            dataframes.append(df)

        except Exception as e:
            print(f"Error processing {item}: {e}")
            continue

    # If no valid dataframes were collected, return an empty DataFrame
    if not dataframes:
        print("No valid data found.")
        return pd.DataFrame()

    # Concatenate all DataFrames in the list to create a single DataFrame
    final_df = pd.concat(dataframes, ignore_index=True)

    return final_df

# Feature info endpoint
@app.get("/feature_info/{feature_name}")
async def feature_info(feature_name: FeatureInfo):
    """
    Retrieve information about a feature.
    """
    try:
        info = examples['features_info'][feature_name]
        return info
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found.")

# Prediction endpoint
@app.post("/predict/")
async def predict(
    categorical_features: List[str] = categorical_features,
    label: str = label,
    example: str = "Class <=50k (Label 0)",  # Default example type
    example_data: dict = Body(...),  # Receive the example data as JSON
):
    """
    Predict the salary based on the provided example data.
    This endpoint processes the input data and returns the prediction result.
    """
    # Extract the data from the provided example data
    df = example_data_extract(example_data)

    # Process the data (this will return X and y)
    X, y = process_data(df, categorical_features, label)

    # Filter the data to match the selected example type (e.g., "Class <=50k (Label 0)")
    X_example = X[X["example_type"] == example].drop("example_type", axis=1)

    # Perform inference (model inference using the X_example data)
    pred_label, pred_prob = inference(model, X_example)

    # Convert predicted label to integer and probability to float
    pred_label = int(pred_label)
    pred_probs = float(pred_prob[:, 1])

    # Determine salary class based on the predicted label
    pred_salary = ">50k" if pred_label == 1 else "<=50k"

    # Return the prediction results
    return {
        'label': pred_label,
        'prob': pred_probs,
        'salary': pred_salary
    }
