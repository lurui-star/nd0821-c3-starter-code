"""
Author: Rui Lu
Date: December, 2024
This script holds main functions for fastapi app
"""
import os
import yaml
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from pipeline.data import process_data
from pipeline.model import inference
from app.schemas import FeatureInfo, ExampleType

# Define the application
app = FastAPI(
    title="Udacity - Project 3",
    description="Deploying a Machine Learning Model on Heroku with FastAPI",
    version="0.1",
)

with open("/Users/ruilu/nd0821-c3-starter-code/starter/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Model path setup
model = joblib.load(config['model_dir'])

# Feature and Example Information
with open(config['example_dir']) as fp:
    examples = yaml.safe_load(fp)

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
        ExampleType.class_less_than_50k,
        ExampleType.class_greater_than_50k,
        ExampleType.missing_sample,
        ExampleType.error_sample
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

            # Add the 'example_type' to keep track of the source
            df['example_type'] = item.value

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
async def feature_info(feature_name: str):
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
    example: str = ExampleType.class_less_than_50k,  # Default example type
):
    # Extract example data into DataFrame
    df = example_data_extract(examples)

    if df.empty:
        raise HTTPException(status_code=404, detail="No valid example data available.")

    # Process the data (this will return X and y)
    X, y = process_data(df, config["main"]["data"]["categorical_features"],config["main"]["data"]["label"])
    
    # Filter the data to match the selected example type (e.g., "Class <=50k (Label 0)")
    X_example = X[X["example_type"] == example].drop("example_type", axis=1)

    if X_example.empty:
        raise HTTPException(status_code=404, detail=f"No data available for example type: {example}")

    # Perform inference (model inference using the X_example data)
    pred_label, pred_prob = inference(model, X_example)
    
    # Convert predicted label to integer and probability to float
    pred_label = int(pred_label)
    pred_prob = float(pred_prob[:, 1])

    # Determine salary class based on the predicted label
    pred_salary = ">50k" if pred_label == 1 else "<=50k"

    # Return the prediction results
    return {
        'label': pred_label,
        'prob': pred_prob,
        'salary': pred_salary
    }
