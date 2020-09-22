import logging
import random
import joblib 

from fastapi import APIRouter
import pandas as pd

log = logging.getLogger(__name__)
router = APIRouter()
log_reg = joblib.load('app/api/log_reg.joblib')
print('pickle model loaded!')

@router.get('/random')
def predict():
    """Returns a random true or false value"""
    return random.choice(['Succeeded', 'Fail'])