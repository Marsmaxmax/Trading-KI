import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.customError import Profitloss
from util.customfunctions import load_custom_model
from util.datasequencer import create_sequences
import os
import sys


input_file = 'data/output.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
batch = 4
runs = int()

tf.debugging.set_log_device_placement(False)
# sys.argv[0] ist der Name des Skripts
# sys.argv[1] ist das erste Argument von der Kommandozeile
if len(sys.argv) > 1:
    print(f" Anzahl Durchläufe {sys.argv[1]}")
    runs = int(sys.argv[1])
elif len(sys.argv) > 2:
    print(f" Batch Anzahl{sys.argv[2]}")
    batch = int(sys.argv[2])
else:
    print("Keine Argumente eingegeben.")
    exit()

data = pd.read_csv(input_file, header=None)
candles = data.values
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_candle = create_sequences(candles, 64)

model = load_custom_model(model_file)


for i in range((runs//50) + 1 ):
    history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], y_candle, epochs=50, batch_size=batch, validation_split=0.2)
    model.save(f'history/trend_model_epoch{(i+1)*50}.keras')
    

model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')