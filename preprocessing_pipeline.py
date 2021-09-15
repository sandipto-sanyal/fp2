# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 14:26:23 2021

@author: sandipto.sanyal
"""

import pandas as pd
import constants as c
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
def prediction_preprocess(df:pd.DataFrame,
                          dict_of_transformers: dict
                          ):
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
       
    X_test:
        The X Tensor to be predicted by model
    '''     
    
    # rearrange the columns
    
    test_df = df[c.ohe_cols+ \
                       c.cv_cols+ \
                       c.numeric_cols
                       ]
    
    X_test = np.empty(shape=(len(test_df),0))
    
    # perform one hot encoding
    X_test = np.hstack((X_test,dict_of_transformers['one_hot_encoder'].transform(test_df[c.ohe_cols])))
    
    # create the count vectorizers
    for col in c.cv_cols:
        # fill na
        test_df[col] = test_df[col].fillna('unk')
        regex = '\, '
        cv = CountVectorizer(tokenizer=lambda text: re.split(regex,text), 
                             binary=True, 
                             min_df=10, 
                             vocabulary=dict_of_transformers[col]
                             )
        
        X_test_temp = cv.transform(test_df[col]).toarray()
        X_test = np.hstack((X_test,X_test_temp))
    
    # impute cost column
    X_test = np.hstack((X_test,test_df['cost'].values.reshape(-1,1)))
    X_test = dict_of_transformers['imputer'].transform(X_test)
    
    # scale the cost
    X_test[:,-1] = dict_of_transformers['cost_scaler'].transform(X_test[:,-1].reshape(-1,1)).reshape(-1,)
    
    # return
    return X_test