"""
Author: Rui Lu
Date: December, 2024
This script used for run training, evaluting and saving the model
"""

import sys
import joblib
import logging
from sklearn.model_selection import train_test_split
from pipeline import data
from pipeline import evaluate
from pipeline import save_output
from pipeline import train
from pipeline import utils
import config


logging.basicConfig(stream=sys.stdout, level=logging.INFO)

