import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from util.datasequencer import create_predsequences, create_sequences

# Parameter
input_file = 'data/predict.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
seq_length = 100

# CSV-Datei laden
data = pd.read_csv(input_file, header=None)
candles = data.values

x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_candle = create_sequences(candles, 64)
# Modell laden
model = load_model(model_file)
print(f'Modell "{model_file}" erfolgreich geladen.')
print(x_close.shape)
# Vorhersagen treffen
predictions = model.predict([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3])


# Beispielausgabe
print(x_close[0, 63])
print(y_candle)
print(predictions)
