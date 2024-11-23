import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from util.customError import Profitloss
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os
import sys

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
data = pd.read_csv(input_file, header=None)
candles = data.values  # Close, Open, High, Low
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_candle = create_sequences(pd.read_csv(input_file, header=None).values, 64)
# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()


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

model.compile(optimizer=customoptimizer, loss=Profitloss(0.1), metrics=['mse','mae','accuracy'])
# history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], y_candle, epochs=1, batch_size=1, validation_split=0.2)
model.save(model_file)