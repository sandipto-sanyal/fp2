# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:44:05 2021
@author: sandipto.sanyal
"""

# column selectors
ohe_cols = ['online_order', 
           'book_table', 
           'listed_in_city', 
           'listed_in_type'
          ]
cv_cols = ['cuisines', 
           'rest_type'
          ]
numeric_cols = ['cost']
y = ['rate']


# output columns
output_cols = ['name',
               'predicted_rating',
               'address',
               'cuisines',
               'cost',
               'menu_item'
               ]

# gcloud model prediction constants
project_id = 'sandipto-project'
bucket_name = 'foundation_project2'
model_dir = 'bin_files/model'
transformers_dir = 'bin_files/transformers'
model_name = 'food_model'
prediction_folder = 'gs://foundation_project2/prediction_folder'
