import pandas as pd
import numpy as np
import tensorflow as tf
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

customoptimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9,clipnorm=1.0)

data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_close, y_open, y_high, y_low = create_sequences(candles, 64)

model.compile(optimizer=customoptimizer,loss={'output_close': 'binary_crossentropy', 'output_open': 'binary_crossentropy', 'output_high': 'binary_crossentropy', 'output_low': 'binary_crossentropy'}, metrics={'output_close': 'accuracy', 'output_open': 'accuracy', 'output_high': 'accuracy', 'output_low': 'accuracy'})
history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], [y_close, y_open, y_high, y_low], epochs=20, batch_size=16, validation_split=0.2)
model.save(model_file)