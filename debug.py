# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:24:50 2021

@author: sandipto.sanyal
"""

import tensorflow as tf
path = r'C:\Users\sandipto.sanyal\Documents\bin_files\model\v20210917134450'
model = tf.keras.models.load_model(path)
tf.keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,

)
