# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:46:45 2021

@author: sandipto.sanyal
"""

import tensorflow as tf
import constants as c
import numpy as np

def create_model_architecture(X_train):
    '''
    Creates the model architecture

    Parameters
    ----------
    X_train : np.ndarray of shape (samples, dimension)

    Returns
    -------
    model : tf.keras.Model
        The compiled model

    '''
    input_layer = tf.keras.Input(shape=(X_train.shape[1],), 
                                 name='input_layer', 
                                 dtype='float32')
    h1 = tf.keras.layers.Dense(16, activation='relu', name='h1')(input_layer)
    outputs = tf.keras.layers.Dense(1, activation='relu', name='output')(h1)
    
    model = tf.keras.Model(inputs=input_layer,outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer,loss=loss)
    
    return model

if __name__ == '__main__':
    X_train = np.array([[1,2,3],[4,5,6]])
    model = create_model_architecture(X_train)
    