import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os
import sys


input_file = 'data/output.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
batch = 32
runs = int()

tf.debugging.set_log_device_placement(False)

# sys.argv[0] ist der Name des Skripts
# sys.argv[1] ist das erste Argument von der Kommandozeile
if len(sys.argv) > 1:
    print(f" Anzahl Durchläufe{sys.argv[1]}")
    runs = int(sys.argv[1])
    batch = 32
elif len(sys.argv) > 2:
    print(f" Batch Anzahl{sys.argv[2]}")
    batch = int(sys.argv[2])
else:
    print("Keine Argumente eingegeben.")
    exit()





data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_close, y_open, y_high, y_low = create_sequences(candles, 64)

# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()

history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], [y_close, y_open, y_high, y_low], epochs=runs, batch_size=batch, validation_split=0.2)

model.summary()

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')