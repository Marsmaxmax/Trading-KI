import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Parameter
input_file = 'predict.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird
seq_length = 100

# CSV-Datei laden
data = pd.read_csv(input_file, header=None)
candles = data.values  # Open, Close, High, Low

# Überprüfen der Anzahl der Einträge
if len(candles) <= seq_length:
    raise ValueError(f"Nicht genügend Datenpunkte, um eine Sequenz der Länge {seq_length} zu erstellen. "
                     f"Es werden mindestens {seq_length + 1} Datenpunkte benötigt.")

# Funktion zur Generierung von Eingabesequenzen
def create_sequences(candles, seq_length):
    X = []
    for i in range(len(candles) - seq_length):
        seq = candles[i:i + seq_length]
        X.append(seq)
    return np.array(X)

# Daten in Sequenzen umwandeln
X = create_sequences(candles, seq_length)

# Daten skalieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

# Modell laden
model = load_model(model_file)
print(f'Modell "{model_file}" erfolgreich geladen.')

# Vorhersagen treffen
up_probabilities, down_probabilities = predictions = model.predict(X_scaled)


# Beispielausgabe
for i in range(5):
    print(f"Aufwärtstrend Wahrscheinlichkeit: {up_probabilities[i][0]:.2f}, Abwärtstrend Wahrscheinlichkeit: {down_probabilities[i][0]:.2f}")
