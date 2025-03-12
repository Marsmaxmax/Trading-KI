import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from config import INPUT_LENGTH
from preperation.datasequencer import create_sequences

# Parameter
input_file = 'data/XAUUSD/predict.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
seq_length = 100

# CSV-Datei laden
data = pd.read_csv(input_file, header=None)
candles = data.values

x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)
# Modell laden
model = load_model(model_file)
print(f'Modell "{model_file}" erfolgreich geladen.')
# Vorhersagen treffen
predictions = model.predict([x_candle, x_ema])


# Beispielausgabe
print(y_direction)
print(y_long)
print(y_short)
print(predictions)
