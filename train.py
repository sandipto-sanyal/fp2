# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:24:33 2021

@author: sandipto.sanyal
"""

import constants as c
import preprocessing_pipeline
import model_architecture
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
import pickle as pkl
import numpy as np
import os
from datetime import datetime
import shutil
import argparse



class Train:
    def __init__(self, df_path,
                 epochs:int
                 ):
        self.df = pd.read_csv(df_path)
        self.epochs = epochs
        self.create_directories()
    
    def create_directories(self):
        try:
            os.mkdir(c.bin_files_path)
            os.mkdir(c.transformer_folder)
        except FileExistsError:
            shutil.rmtree(c.bin_files_path)
            os.mkdir(c.bin_files_path)
            os.mkdir(c.transformer_folder)
            
        
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
        self.dict_of_transformers, \
        self.X_train, \
        self.y_train, \
        self.X_test, \
        self.y_test = preprocessing_pipeline.training_preprocess(self.df)
        self.model = model_architecture.create_model_architecture(self.X_train)
        
        callbacks = [tf.keras.callbacks.EarlyStopping(patience=10,verbose=1, restore_best_weights=True)]
        history = self.model.fit(x=self.X_train,y=self.y_train, 
                          batch_size=32,
                          validation_data=(self.X_test,self.y_test),
                          validation_batch_size=32,
                          epochs=self.epochs,
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
            with open(os.path.join(c.transformer_folder,cv_col),'wb') as f:
                pkl.dump(self.dict_of_transformers[cv_col].vocabulary_, f)
                # remove the element
                self.dict_of_transformers.pop(cv_col)
        
        ## for the rest of the transformers save the transformer
        key_transformer_list = list(self.dict_of_transformers.items())
        for key, transformer in key_transformer_list:
            with open(os.path.join(c.transformer_folder,key),'wb') as f:
                pkl.dump(transformer,f)
                # remove the element
                self.dict_of_transformers.pop(key)
        # model
        self.model.save(c.model_path)
        print('Save successful')
    
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
    
    def upload_binaries(self):
        '''
        Uploads the model binaries

        Returns
        -------
        None.

        '''
        command = 'gsutil -m cp -r {} {}'.format(c.bin_files_path, c.gcs_path)
        os.system(command)
        # delete intermediate files
        shutil.rmtree(c.bin_files_path)
        
    
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
                  f'--origin={c.model_dir} ' \
                  '--runtime-version=2.3 ' \
                  f'--framework={c.framework} ' \
                  '--python-version=3.7 ' \
                  f'--region={c.region} ' \
                  f'--machine-type={c.machine_type} ' \
                  f'--description=r2_score={self.r2_score} ' \
                  '--max-nodes=1 --min-nodes=1'
        os.system(command)
        print('Check: {} for model version: {}'.format(c.website_path, self.version))
 
def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
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
     
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    tr = Train(df_path=args.df_path, epochs=args.epochs)
    tr.train()
    tr.export_binaries()
    tr.evaluate()
    tr.upload_binaries()
    tr.deploy_model()