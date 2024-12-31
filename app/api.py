import os
import yaml
import joblib
import numpy as np
from fastapi import FastAPI, Body, HTTPException
from enum import Enum
from typing import Optional
from pydantic import BaseModel

# Define the application
app = FastAPI(
    title="Udacity - Project 3",
    description="Deploying a Machine Learning Model on Heroku with FastAPI",
    version="0.1",
)

# Model path setup
MODEL_PATH = os.path.join(
    "/Users/ruilu/nd0821-c3-starter-code/starter/model", "best_model.pkl"
)
model = joblib.load(MODEL_PATH)

# Feature and Example Information
EXAMPLES_PATH = "/Users/ruilu/nd0821-c3-starter-code/starter/app/examples.yaml"
with open(EXAMPLES_PATH) as fp:
    examples = yaml.safe_load(fp)

# Pydantic model for input validation
class Person(BaseModel):
    age: int
    workclass: Optional[str] = None
    fnlgt: int
    education: Optional[str] = None
    education_num: int
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Optional[str] = None

# Enum for feature names
class FeatureInfo(str, Enum):
    age = "age"
    workclass = "workclass"
    fnlgt = "fnlgt"
    education = "education"
    education_num = "education_num"
    marital_status = "marital_status"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    capital_gain = "capital_gain"
    capital_loss = "capital_loss"
    hours_per_week = "hours_per_week"
    native_country = "native_country"

# Greeting endpoint
@app.get("/")
async def greetings():
    return {"greeting": "Welcome to salary prediction!"}

# Feature info endpoint
@app.get("/feature_info/{feature_name}")
async def feature_info(feature_name: FeatureInfo):
    try:
        info = examples['features_info'][feature_name]
        return info
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Feature '{feature_name}' not found.")

# Prediction endpoint
@app.post("/predict/")
async def predict(person: Person = Body(..., examples=examples['post_examples'])):
    person_dict = person.dict()

    # Extract features from person data
    try:
        features = np.array([person_dict[f] for f in examples['features_info'].keys()]).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    # Create DataFrame with appropriate columns
    df = pd.DataFrame(features, columns=examples['features_info'].keys())

    # Make prediction using the model
    pred_label = int(model.predict(df))
    pred_probs = float(model.predict_proba(df)[:, 1])
    
    # Determine salary class
    pred_salary = ">50k" if pred_label == 1 else "<=50k"

    return {
        'label': pred_label,
        'prob': pred_probs,
        'salary': pred_salary
    }
