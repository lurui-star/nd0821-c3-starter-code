"""
Author: Rui Lu 
Date: December, 2024
This script holds the test functions for api module
"""
import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_greetings():
    """
    Tests GET greetings function
    """
    response = client.get('/')
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"
    assert response.json() == {"greeting": "Welcome to salary prediction!"}


@pytest.mark.parametrize('test_input, expected', [
    ('age', "Age of the person - numerical - int"),
    ('fnlgt', 'MORE INFO NEEDED - numerical - int'),
    ('race', 'Race of the person - nominal categorical - str')
])
def test_feature_info_status(test_input: str, expected: str):
    """
    Tests GET request and status code for feature_info function

    Args:
        test_input (str): example input
        expected (str): example output
    """
    response = client.get(f'/feature_info/{test_input}')
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "GET"


@pytest.mark.parametrize('test_input, expected', [
    ('age', "Age of the person - numerical - int"),
    ('fnlgt', 'MORE INFO NEEDED - numerical - int'),
    ('race', 'Race of the person - nominal categorical - str')
])
def test_feature_info_response(test_input: str, expected: str):
    """
    Tests GET request response for feature_info function

    Args:
        test_input (str): example input
        expected (str): example output
    """
    response = client.get(f'/feature_info/{test_input}')
    assert response.json() == expected


def test_predict_status():
    """
    Tests POST predict function status
    """
    data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/predict/", json=data)
    assert response.status_code == HTTPStatus.OK
    assert response.request.method == "POST"


def test_predict_response():
    """
    Tests POST request response for predict function when successful against a sample
    """
    data = {
        'age': 38,
        'fnlgt': 15,
        'education_num': 1,
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 5
    }
    response = client.post("/predict/", json=data)
    assert response.json()['label'] == 0 or response.json()['label'] == 1
    assert response.json()['prob'] >= 0 and response.json()['label'] <= 1
    assert response.json()['salary'] == '>50k' or response.json()[
        'salary'] == '<=50k'
