import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os
import sys

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()


def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)

customoptimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,           
    momentum=0.8,      
    epsilon=1e-7,      
    centered=True,      
    clipnorm=1.0,       
    clipvalue=None,       
    global_clipnorm=None 
    )

model.compile(optimizer=customoptimizer, loss='mse', metrics=['mse','mae','accuracy'])
model.save(model_file)