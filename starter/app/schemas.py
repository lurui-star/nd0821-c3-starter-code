"""
Author: Rui Lu
Date: December, 2024
This script holds schemas for fastapi app
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel

# Define the Enum for valid example types


class ExampleType(str, Enum):
    class_less_than_50k = "Class <=50k (Label 0)"
    class_greater_than_50k = "Class >50k (Label 1)"
    missing_sample = "Missing sample"
    error_sample = "Error sample"


# Pydantic model for input validation
class FeatureInfo(str, Enum):
    age = "age"
    workclass = "workclass"
    fnlgt = "fnlgt"
    education = "education"
    education_num = "education-num"
    marital_status = "marital-status"
    occupation = "occupation"
    relationship = "relationship"
    race = "race"
    sex = "sex"
    capital_gain = "capital-gain"
    capital_loss = "capital-loss"
    hours_per_week = "hours-per-week"
    native_country = "native-country"


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
    country: Optional[str] = None
