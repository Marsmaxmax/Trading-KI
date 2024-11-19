import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from util.datasequencer import create_predsequences

# Parameter
input_file = 'predict.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
seq_length = 100

# CSV-Datei laden
data = pd.read_csv(input_file, header=None)
candles = data.values

x = {x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3} = create_predsequences(candles, 64)
# Modell laden
model = load_model(model_file)
print(f'Modell "{model_file}" erfolgreich geladen.')

# Vorhersagen treffen
up_probabilities, down_probabilities = predictions = model.predict(x)


# Beispielausgabe
for i in range(5):
    print(f"Aufwärtstrend Wahrscheinlichkeit: {up_probabilities[i][0]:.2f}, Abwärtstrend Wahrscheinlichkeit: {down_probabilities[i][0]:.2f}")
