# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:44:05 2021
@author: sandipto.sanyal
"""
# training path
training_file_cloud = 'gs://foundation_project2/training_folder/zomato.csv'
training_file_local = '../datasets/Zomato Bangalore Restaurants Data/zomato.csv'
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

# training hyper params
test_size = 0.3
epochs = 1

# model export paths
model_path = 'model'
transformer_folder = 'transformers'
gcs_path = 'gs://foundation_project2'

# gcloud model deployment constants
model_dir = 'gs://foundation_project2/bin_files/model'
model_name = 'food_model'
framework = 'TENSORFLOW'
region = 'us-central1'
machine_type = 'n1-standard-2'
website_path = 'https://console.cloud.google.com/ai-platform/models?'

# bigquery paths
table_id='sandipto-project.foodrecommender.model_version'