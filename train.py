import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from config import INPUT_LENGTH
from util.custom.customfunctions import load_custom_model
from preperation.datasequencer import create_sequences
import os
import sys


input_file = 'data/XAUUSD/output.csv'  # Name der Eingabedatei
training_set_1 = 'data/XAUUSD/train_1.csv'
training_set_2 = 'data/XAUUSD/train_2.csv'
training_set_3 = 'data/XAUUSD/train_3.csv'
training_set_4 = 'data/XAUUSD/train_4.csv'
training_sets = [training_set_1, training_set_2, training_set_3, training_set_4]
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
batch = 1
runs = int()

tf.debugging.set_log_device_placement(False)
# sys.argv[0] ist der Name des Skripts
# sys.argv[1] ist das erste Argument von der Kommandozeile
if len(sys.argv) > 1:
    print(f" Anzahl Durchläufe {sys.argv[1]}")
    runs = int(sys.argv[1])
# elif len(sys.argv) > 2:
#     print(f" Batch Größe{sys.argv[2]}")
#     batch = int(sys.argv[2])
else:
    print("Keine Argumente eingegeben.")
    exit()

data = pd.read_csv(input_file, header=None)
candles = data.values
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)
x_candle = np.transpose(x_candle, (1, 2, 0))
x_ema = np.transpose(x_ema, (1, 2, 0))
y_direction = np.transpose(y_direction, (1, 0))
y_long = np.transpose(y_long, (1, 0))
y_short = np.transpose(y_short, (1, 0))

model = load_custom_model(model_file)
if runs < 50 :
    history = model.fit([x_candle, x_ema], [y_long, y_short], epochs = runs, batch_size=batch, validation_split=0.1)
# else:
#     for i in range((runs//50) + 1 ):
#         selected_training_set = random.choice(training_sets)
#         print(f"Verwende Datensatz: {selected_training_set}")
#         x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y = create_sequences(pd.read_csv(selected_training_set, header=None).values, 64)
#         x = [x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3]
#         history = model.fit(x, y, epochs=50, batch_size=batch, validation_split=0.1)
#         model.save(f'history/trend_model_epoch{(i+1)*50}.keras')
    

model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')