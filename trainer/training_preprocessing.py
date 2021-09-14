# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 09:58:25 2021

@author: sandipto.sanyal
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import numpy as np
import re
import constants as c

     

def prediction_preprocess(df:pd.DataFrame):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    pass

def training_preprocess(df:pd.DataFrame):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame must mandatorily contain the following schema.
        ['url', # URL for the restaurants
         'address', # address of the restaurant
         'name', # String  
         'online_order', # Yes/ No 
         'book_table', # Yes/ #No
         'rate', # Float between 0 to 5
         'votes', # Integers
         'phone', # String
         'location', # String
         'rest_type', # ', ' separated vals
         'dish_liked', # ', ' separated vals
         'cuisines',  # ', ' separated vals
         'cost', # Integer 
         'reviews_list', # list of reviews Optional
         'menu_item', # ', ' separated vals
         'listed_in_type', # String
         'listed_in_city' # String
         ]

    Returns
    -------
    dict_of_transformers: The transformation in dict
    X_train:
    y_train:
    X_test:
    y_test:
        DESCRIPTION.

    '''     
    # rearrange the columns
    
    df_rearranged = df[c.ohe_cols+ \
                       c.cv_cols+ \
                       c.numeric_cols+ \
                       c.y
                       ]
    train_df, test_df = train_test_split(df_rearranged, test_size=c.test_size)
    
    # initialize all
    X_train = np.empty(shape=(len(train_df),0))
    X_test = np.empty(shape=(len(test_df),0))
    dict_of_transformers = {}
    
    # create the one hot encoder
    transformer_type = 'one_hot_encoder'
    dict_of_transformers[transformer_type] = OneHotEncoder(drop='first', sparse=False)
    X = dict_of_transformers[transformer_type].fit_transform(train_df[c.ohe_cols])
    X_train = np.hstack((X_train,X))
    X_test = np.hstack((X_test,dict_of_transformers[transformer_type].transform(test_df[c.ohe_cols])))
    
    # create the count vectorizers
    for col in c.cv_cols:
        # fill na
        train_df[col] = train_df[col].fillna('unk')
        test_df[col] = test_df[col].fillna('unk')
        regex = '\, '
        dict_of_transformers[col] = CountVectorizer(tokenizer=lambda text: re.split(regex,text), binary=True, min_df=10)
        X = dict_of_transformers[col].fit_transform(train_df[col])
        X = X.toarray()
        
        
        X_train = np.hstack((X_train,X))
        X_test_temp = dict_of_transformers[col].transform(test_df[col]).toarray()
        X_test = np.hstack((X_test,X_test_temp))
        
    # impute the cost column
    X_train = np.hstack((X_train,train_df['cost'].values.reshape(-1,1)))
    X_test = np.hstack((X_test,test_df['cost'].values.reshape(-1,1)))
    
    dict_of_transformers['imputer'] = KNNImputer(n_neighbors=5)
    X_train = dict_of_transformers['imputer'].fit_transform(X_train)
    X_test = dict_of_transformers['imputer'].transform(X_test)
    
    # scale the cost
    dict_of_transformers['cost_scaler'] = StandardScaler()
    dict_of_transformers['cost_scaler'].fit(X_train[:,-1].reshape(-1,1))
    X_train[:,-1] = dict_of_transformers['cost_scaler'].transform(X_train[:,-1].reshape(-1,1)).reshape(-1,)
    X_test[:,-1] = dict_of_transformers['cost_scaler'].transform(X_test[:,-1].reshape(-1,1)).reshape(-1,)
    
    
    # get the ouput columns
    y_train = train_df['rate'].values
    y_test = test_df['rate'].values
    
    # return all the things
    return dict_of_transformers, X_train, y_train, X_test, y_test

if __name__ == '__main__':
    df = pd.read_csv('../datasets/Zomato Bangalore Restaurants Data/zomato.csv')
    dict_of_transformers, X_train, y_train, X_test, y_test = training_preprocess(df)