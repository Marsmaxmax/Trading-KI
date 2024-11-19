import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os
import sys

model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird

# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()

customoptimizer = tf.keras.optimizers.RMSprop()

model.compile(optimizer=customoptimizer,loss={'output_close': 'binary_crossentropy', 'output_open': 'binary_crossentropy', 'output_high': 'binary_crossentropy', 'output_low': 'binary_crossentropy'}, metrics={'output_close': 'accuracy', 'output_open': 'accuracy', 'output_high': 'accuracy', 'output_low': 'accuracy'})
model.save(model_file)