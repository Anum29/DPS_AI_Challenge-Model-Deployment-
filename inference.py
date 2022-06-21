import sys
import os
import subprocess
import json
import requests
import pandas as pd
import numpy as np
import datetime
import calendar
import re
import time
import json
import boto3
import math
import pickle
import sklearn
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
s3 = boto3.resource('s3')

from sklearn.linear_model import Lasso
import logging
logger = logging.getLogger(__name__)

scalerin = pickle.loads(s3.Bucket('dps-ai-challenge-model').Object("model-weights/scaler.sav").get()['Body'].read())
scalerout = pickle.loads(s3.Bucket('dps-ai-challenge-model').Object("model-weights/out_scaler.sav").get()['Body'].read())

def model_fn(model_dir):
    
    #rng = np.random.RandomState(0)
    #model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    
    logger.info('Done loading model')
    model = pickle.loads(s3.Bucket('dps-ai-challenge-model').Object("model-weights/model.sav").get()['Body'].read())
    return model

def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')

    prediction = model.predict(input_data)  
    return prediction

def input_fn(request_body, content_type='application/json'):
    
    if content_type == 'application/json':
        input_data = json.loads(request_body)       
        year = int(input_data['year'])
        month =int(input_data['month'])
        data = [year, month]
        
        # pre process data above this line and make 'data' list
        #return data
        return scalerin.transform([data])
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    output = scalerout.inverse_transform(prediction_output.reshape(-1,1))
    #make output json    
    prediction = {"prediction" : str(int(output))}  
    
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')

# input json    
# {
#     "year":"2020",
#     "month":"9",
#     "week":"46",
#     "get_revenue":"5637222733",
#     "fb_total_revenue":"444332"
# }
#output json
# {
#     "fb_total_spend" :"7384"
# }