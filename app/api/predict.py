import logging
import random
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi import APIRouter
from typing import Dict

from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import OneHotEncoder, OrdinalEncoder


log = logging.getLogger(__name__)
router = APIRouter()
#log_reg = joblib.load('app/api/log_reg.joblib')
#print('pickle model loaded!')
df = pd.read_csv('app/cleaned_kickstarter_data.csv')

@router.post('/predict')
def predict(user_input: Dict):
    user_input = create_df(user_input)
    """Returns a random true or false value"""
    train, test = train_test_split(df, train_size=0.80, test_size=0.20, 
                                    stratify=df['project_success'], random_state=42)
    # select our target 
    target = 'project_success'

    # make train without our target or id
    train_features = train.drop(columns=[target])

    # make numeric features
    numeric_features = train_features.select_dtypes(include='number').columns.tolist()

    # make a cardinality feature to help filter
    cardinality = train_features.select_dtypes(exclude='number').nunique()

    # get a list of relevant categorical data
    categorical_features = cardinality[cardinality <=50].index.tolist()

    # Combine the lists 
    features = numeric_features + categorical_features

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    # print(features)
    # print(X_train.shape, X_test.shape)

    lrmodel = Pipeline([
                    ('ohe', OneHotEncoder(use_cat_names=True)),
                    ('scaler', StandardScaler()),  
                    ('impute', SimpleImputer()),
                    ('classifier', LogisticRegressionCV())
                    ])
    lrmodel.fit(X_train, y_train)

    row = X_test.iloc[[4]]
    # print(X_train)
    # print('training accuracy:', lrmodel.score(X_train, y_train))
    # print('test accuracy:', lrmodel.score(X_test, y_test))
    # if lrmodel.predict(row) == 1:
    #   return 'Your Kickstarter project is likely to succeed!'
    # else:
    #   return 'Your Kickstarter project is likely to fail.'
    # print(X_test.head())
    # print(user_input)
    # print(y_test.head())
    # print(y_test.iloc[[0]])

    if lrmodel.predict(user_input) == 1:
        return 'Your Kickstarter project is likely to succeed!'
    else:
        return 'Your Kickstarter project is likely to fail.'

def create_df(web_in):
    '''Takes incoming dictionaries and turns it into a pandas dataframe'''
    input_frame = pd.DataFrame(web_in, index=[0])

    # Changing datatype of start and end to date time
    # Adding column length of campaign
    input_frame['deadline'] = pd.to_datetime(input_frame['deadline'])
    input_frame['launched'] = pd.to_datetime(input_frame['launched'])
    input_frame['length_of_campaign'] = (input_frame['deadline'] - input_frame['launched']).dt.days

    # Using a pretrained neural network to encode title to numbers
    # Adding numbers to column as sentiments
    sentiments =[] 
    analyzer = SentimentIntensityAnalyzer()
    for sentence in input_frame['name']:
        vs = analyzer.polarity_scores(sentence)
        sentiments.append(vs['compound'])
    input_frame['sentiments'] = sentiments
    
    # input_frame['goal'] = (input_frame['goal'].str.split()).apply(lambda x: float(x[0].replace(',', '')))
    # input_frame['backers']= input_frame['backers'].astype(str).astype(int)

    # Dropping unecessary username column
    input_frame = input_frame.drop('username', axis=1)
    input_frame = input_frame.drop('name', axis=1)
    input_frame = input_frame.drop('launched', axis=1)
    input_frame = input_frame.drop('deadline', axis=1)

    input_frame = input_frame[['goal', 'backers', 'length_of_campaign', 'sentiments', 'main_category']]

    userinput = input_frame.iloc[[0]]

    return userinput 

    def test():
        input = {"name": "terrible MegaBuster from Megaman X","goal": 10000,
        "launched": "2015-08-11",
        "deadline": "2015-08-18",
        "backers":21,
        "main_category": 11,
        "username": "LoginID"}
        
        user = create_df(input)
        return predict(user)