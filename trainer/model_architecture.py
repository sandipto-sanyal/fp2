# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:46:45 2021

@author: sandipto.sanyal
"""

import tensorflow as tf
from . import constants as c
import numpy as np

def create_model_architecture(X_train,
                              hidden_layers:int=1,
                              units:int=16
                              ):
    '''
    Creates the model architecture

    Parameters
    ----------
    X_train : np.ndarray of shape (samples, dimension)
    hidden_layers : int (optional) number of hidden layers the model to have default=1
    units: int (optional) number of units in each hidden layer the model to have default=16

    Returns
    -------
    model : tf.keras.Model
        The compiled model

    '''
    input_layer = tf.keras.Input(shape=(X_train.shape[1],), 
                                 name='input_layer', 
                                 dtype='float32')
    for h in range(1, hidden_layers+1):
        if h == 1:
            h1 = tf.keras.layers.Dense(units, activation='relu', name='h{}'.format(h))(input_layer)
        else:
            h1 = tf.keras.layers.Dense(units, activation='relu', name='h{}'.format(h))(h1)
    outputs = tf.keras.layers.Dense(1, activation='relu', name='output')(h1)
    
    model = tf.keras.Model(inputs=input_layer,outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer,loss=loss,metrics=[tf.keras.metrics.MeanSquaredError()],)
    
    return model

if __name__ == '__main__':
    X_train = np.array([[1,2,3],[4,5,6]])
    model = create_model_architecture(X_train)
    