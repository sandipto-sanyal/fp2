# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:33:27 2021

@author: sandipto.sanyal
"""

from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
from google.cloud import storage

from tensorflow.python.lib.io import file_io
import tensorflow as tf
import pickle as pkl
import pandas as pd
import os

import constants as c
import preprocessing_pipeline

class Predict:
    def __init__(self, 
                 model_version: str,
                 area_name: str
                 ):
        '''
        

        Parameters
        ----------
        model_version : str
            The model version name for prediction
            
        area_name : str
            The area name from where the request is coming

        Returns
        -------
        None.

        '''
        self.ml = discovery.build('ml','v1')
        self.project_id = 'projects/{}'.format(c.project_id)
        self.request_dict = {'name': c.model_name,
               'description': 'Food recommendation model'}
        self.model_version = model_version
        self.area_name = area_name
        self.storage_client = storage.Client()
        
    
    def open_pickle_files_cloud(self):
        '''
        Opens the model dependencies

        Returns
        -------
        None.

        '''
        bucket = self.storage_client.bucket(c.bucket_name)
        ohe_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'one_hot_encoder'
                                     )
        cv_dictionary_paths = ['{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     element
                                     )
                               for element in c.cv_cols
                               ]
        imputer_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'imputer'
                                     )
        scaler_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'cost_scaler'
                                     )
        self.dict_of_transformers = {}
        # load the OHE
        blob = bucket.blob(ohe_path)
        self.dict_of_transformers['one_hot_encoder'] = pkl.loads(blob.download_as_string())
        
        # load the count vectorizer dictionaries
        for cv_col, cv_dictionary_path in zip(c.cv_cols, cv_dictionary_paths):
            blob = bucket.blob(cv_dictionary_path)
            self.dict_of_transformers[cv_col] = pkl.loads(blob.download_as_string())
        
        # load the imputer
        blob = bucket.blob(imputer_path)
        self.dict_of_transformers['imputer'] = pkl.loads(blob.download_as_string())
        
        # load the scaler
        blob = bucket.blob(scaler_path)
        self.dict_of_transformers['cost_scaler'] = pkl.loads(blob.download_as_string())
    
    def load_data(self):
        '''
        Load and convert the dataset of the given area

        Returns
        -------
        None.

        '''
        dataset_path = '{}/{}.csv'.format(c.prediction_folder,self.area_name)
        self.df = pd.read_csv(dataset_path)
        self.X_test = preprocessing_pipeline.prediction_preprocess(self.df,
                                                                   self.dict_of_transformers
                                                                   )
    
    
    def convert_data_for_prediction_cloud(self):
        '''
        This will convert the tensor to following format as required
        by online prediction routine
        {"instances": [
          {"values": [1, 2, 3, 4], "key": 1},
          {"values": [5, 6, 7, 8], "key": 2}
        ]}

        Returns
        -------
        None.

        '''
        self.instances = {'instances':[]}
        for index, element in enumerate(self.X_test):
            temp_dict = {'values':list(element), 'key':index}
            self.instances['instances'].append(temp_dict)
        pass
    
    def open_pickle_files_local(self):
        '''
        Opens the model dependencies

        Returns
        -------
        None.

        '''
        ohe_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'one_hot_encoder'
                                     )
        cv_dictionary_paths = ['{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     element
                                     )
                               for element in c.cv_cols
                               ]
        imputer_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'imputer'
                                     )
        scaler_path = '{}/{}/{}'.format(c.transformers_dir,
                                     self.model_version,
                                     'cost_scaler'
                                     )
        self.dict_of_transformers = {}
        # load the OHE
        with file_io.FileIO(ohe_path,'rb') as f:
            self.dict_of_transformers['one_hot_encoder'] = pkl.load(f)
        
        # load the count vectorizer dictionaries
        for cv_col, cv_dictionary_path in zip(c.cv_cols, cv_dictionary_paths):
            with file_io.FileIO(cv_dictionary_path,'rb') as f:
                self.dict_of_transformers[cv_col] = pkl.load(f)
        
        # load the imputer
        with file_io.FileIO(imputer_path,'rb') as f:
            self.dict_of_transformers['imputer'] = pkl.load(f)
        
        # load the scaler
        with file_io.FileIO(scaler_path, 'rb') as f:
            self.dict_of_transformers['cost_scaler'] = pkl.load(f)
        
    
    def predict_json(self, project, model, instances, version=None):
        """Send json data to a deployed model for prediction.
    
        Args:
            project (str): project where the AI Platform Model is deployed.
            model (str): model name.
            instances ([Mapping[str: Any]]): Keys should be the names of Tensors
                your deployed model expects as inputs. Values should be datatypes
                convertible to Tensors, or (potentially nested) lists of datatypes
                convertible to tensors.
            version: str, version of the model to target.
        Returns:
            Mapping[str: any]: dictionary of prediction results defined by the
                model.
        """
        service = self.ml
        name = 'projects/{}/models/{}'.format(project, model)

        if version is not None:
            name += '/versions/{}'.format(version)
    
        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()
    
        if 'error' in response:
            raise RuntimeError(response['error'])
    
        return response['predictions']
    
    
    
    def get_predictions_cloud(self):
        '''
        This method gets the version name of the model serving in
        GCP

        Returns
        -------
        None.

        '''
        self.response = self.predict_json(project=c.project_id,
                                     model=c.model_name,
                                     instances=self.instances,
                                     version=self.model_version
                                     )
        
    def get_predictions_with_tf_binaries(self):
        '''
        Since cloud predictions are costly we will perform by loading 
        tensorflow binary files and performing predictions

        Returns
        -------
        None.

        '''
        model_path = f'gs://{c.bucket_name}/{c.model_dir}/{self.model_version}'
        # download to bin_files
        command = 'gsutil -m cp -r {} ./bin_files'.format(model_path)
        os.system(command)
        # model = tf.keras.models.load_model(blob.download_as_string())
        # y_pred = model.predict(self.X_test)
        pass
        
    

if __name__ == '__main__':
    pr = Predict(model_version='v20210914150518',area_name='banashankari')
    pr.open_pickle_files_cloud()
    pr.load_data()
    pr.convert_data_for_prediction_cloud()
    pr.get_predictions_with_tf_binaries()
    