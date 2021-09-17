# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:24:33 2021

@author: sandipto.sanyal

Hyperparameter tuning documentation:
    https://cloud.google.com/ai-platform/training/docs/using-hyperparameter-tuning#gcloud
"""

from . import constants as c
from . import preprocessing_pipeline
from . import model_architecture
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
import pickle as pkl
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import shutil
import argparse
from tensorflow.python.lib.io import file_io
from google.cloud import bigquery



class MyMetricCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('mse', logs['mean_squared_error'], epoch)


class Train:
    def __init__(self, df_path,
                 epochs:int,
                 job_dir:str,
                 args:dict = None
                 ):
        '''
        

        Parameters
        ----------
        df_path : Str
            The full path in where training data is residing in CSV
        epochs : int
            Number of epochs training will run
        job_dir : str
            The path where job will export binaries
        args : dict, optional
            Other arguments to be parsed as deemed essential. The default is None.

        Returns
        -------
        None.

        '''
        self.df = pd.read_csv(df_path)
        self.epochs = epochs
        self.job_dir = job_dir
        self.version = 'v{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        if self.job_dir[0:2] != 'gs':
            # if we are using local paths
            self.create_directories()
        self.args = args
    
    def create_directories(self):
        # creates path recursively as per the folder structure provided
        path_to_create = os.path.join(self.job_dir,c.transformer_folder,self.version)
        Path(path_to_create).mkdir(parents=True,exist_ok=True)
        # try:
        #     os.mkdir(self.job_dir)
        #     os.mkdir(os.path.join(self.job_dir,c.transformer_folder))
        #     os.mkdir(os.path.join(self.job_dir,c.transformer_folder,self.version))
        # except FileExistsError:
        #     shutil.rmtree(self.job_dir)
        #     os.mkdir(self.job_dir)
        #     os.mkdir(os.path.join(self.job_dir,c.transformer_folder))
        #     os.mkdir(os.path.join(self.job_dir,c.transformer_folder,self.version))
            
        
    def train(self):
        '''
        Perform preprocessing of the dataset and model training
    
        Parameters
        ----------
        None
    
        Returns
        -------
        None.
    
        '''
        # hyper parameter tuning code
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()
        
        
        self.dict_of_transformers, \
        self.X_train, \
        self.y_train, \
        self.X_test, \
        self.y_test = preprocessing_pipeline.training_preprocess(self.df)
        self.model = model_architecture.create_model_architecture(self.X_train,
                                                                  hidden_layers=self.args.hidden_layers,
                                                                  units=self.args.units
                                                                  )
        
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10,verbose=1, restore_best_weights=True),
                     MyMetricCallback()
                     ]
        history = self.model.fit(x=self.X_train,y=self.y_train, 
                          batch_size=32,
                          validation_data=(self.X_test,self.y_test),
                          validation_batch_size=32,
                          epochs=self.epochs,
                          verbose=0,
                          callbacks=callbacks
                         )
    
    def export_binaries(self):
        '''
        Exports the binaries and dependencies

        Returns
        -------
        None.

        '''
        # transformers
        ## save the count vectorizer's vocabularies
        for cv_col in c.cv_cols:
            with file_io.FileIO(os.path.join(self.job_dir,c.transformer_folder,self.version,cv_col),'wb') as f:
                pkl.dump(self.dict_of_transformers[cv_col].vocabulary_, f)
                # remove the element
                self.dict_of_transformers.pop(cv_col)
        
        ## for the rest of the transformers save the transformer
        key_transformer_list = list(self.dict_of_transformers.items())
        for key, transformer in key_transformer_list:
            with file_io.FileIO(os.path.join(self.job_dir,c.transformer_folder, self.version,key),'wb') as f:
                pkl.dump(transformer,f)
                # remove the element
                self.dict_of_transformers.pop(key)
        print('Transformers saved at: {}'.format(os.path.join(self.job_dir,c.transformer_folder, self.version)))
        # model
        self.model.save(os.path.join(self.job_dir,c.model_path, self.version))
        print('Model saved successful at: {}'.format(os.path.join(self.job_dir,c.model_path, self.version)))
    
    def evaluate(self):
        '''
        Evaluates the model

        Returns
        -------
        None.

        '''
        y_pred = self.model.predict(self.X_test).reshape(-1,)
        y_pred = np.round(y_pred,decimals=1)
        self.r2_score = r2_score(y_true=self.y_test,y_pred=y_pred)
     
    
    def insert_into_bq(self):
        # Construct a BigQuery client object.
        client = bigquery.Client()

        # TODO(developer): Set table_id to the ID of table to append to.
        # table_id = "your-project.your_dataset.your_table"

        rows_to_insert = [
            {u"version": self.version, u"r2_score": self.r2_score},
        ]

        errors = client.insert_rows_json(c.table_id, rows_to_insert)  # Make an API request.
        if errors == []:
            print("New rows have been added.")
        else:
            print("Encountered errors while inserting rows: {}".format(errors))
    
    
    
    
    def deploy_model(self):
        '''
        Deploys the model in AI Platform

        Returns
        -------
        None.

        '''
        self.version = 'v{}'.format(datetime.now().strftime('%Y%m%d%H%M%S'))
        command = f'gcloud ai-platform versions create {self.version} ' \
                  f'--model={c.model_name} ' \
                   '--async ' \
                  f'--origin={os.path.join(self.job_dir,c.model_path)} ' \
                  '--runtime-version=2.3 ' \
                  f'--framework={c.framework} ' \
                  '--python-version=3.7 ' \
                  f'--region={c.region} ' \
                  f'--machine-type={c.machine_type} ' \
                  f'--description=r2_score={self.r2_score} ' \
                  '--max-nodes=1 --min-nodes=1'
        print('Command:\n{}'.format(command))
        os.system(command)
        print('Check: {} for model version: {}'.format(c.website_path, self.version))
 
def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location where model outputs to be stored '
             )
    
    parser.add_argument(
        '--df-path',
        type=str,
        required=True,
        help='local or GCS location where training dataset is stored '
             )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='number of times to go through the data, default=1')
    
    parser.add_argument(
        '--hidden-layers',
        type=int,
        default=1,
        help='number of hidden layers the model to have default=1')
    
    parser.add_argument(
        '--units',
        type=int,
        default=16,
        help='number of units in each hidden layer the model to have default=16')
     
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tr = Train(df_path=args.df_path, 
               epochs=args.epochs,
               job_dir=args.job_dir,
               args=args
             )
    tr.train()
    tr.export_binaries()
    tr.evaluate()
    if args.job_dir[0:2] == 'gs':
        tr.insert_into_bq()