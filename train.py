import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os
import sys


input_file = 'data/train2.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
runs = int(sys.argv[1])
batch = 32

# sys.argv[0] ist der Name des Skripts
# sys.argv[1] ist das erste Argument von der Kommandozeile
if len(sys.argv) > 1:
    print(f" Anzahl Durchläufe{sys.argv[1]}")
elif len(sys.argv) > 2:
    print(f" Batch Anzahl{sys.argv[2]}")
    batch = int(sys.argv[2])
else:
    print("Keine Argumente eingegeben.")
    exit()





data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100, y_up, y_down = create_sequences(candles)

# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()

model.summary()

y_combined = {'up_trend': y_up, 'down_trend': y_down}

history = model.fit([x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100], y_combined, epochs=runs, batch_size=8, validation_split=0.2)

model.summary()

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')