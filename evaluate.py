import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM
from tensorflow.keras.models import save_model, load_model
import os

# CSV-Datei laden
input_file = 'train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird

# CSV-Datei laden
data = pd.read_csv(input_file, header=None)

# Umwandlung der Werte
candles = data.values  # Open, Close, High, Low

# Funktion zur Generierung von Eingabesequenzen und Zielvariablen
def create_sequences(candles, seq_length=100):
    X, y_up, y_down = [], [], []
    for i in range(len(candles) - seq_length - 1):
        seq = candles[i : i + seq_length]
        X.append(seq)
        
        # Berechnung des Trends
        if candles[i + seq_length + 1][0] > candles[i + seq_length][0]:  # Close der letzten Kerze höher
            y_up.append(1)
            y_down.append(0)
        else:
            y_up.append(0)
            y_down.append(1)
    
    return np.array(X), np.array(y_up), np.array(y_down)

# Daten in Sequenzen umwandeln
X, y_up, y_down = create_sequences(candles)

# Daten aufteilen in Trainings- und Testdaten
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=42
)

# Daten skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
x_test = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

# Überprüfen, ob das Modell existiert und geladen werden kann
if os.path.exists(model_file):
    # Modell laden
    model = load_model(model_file)
    print(f'Modell "{model_file}" erfolgreich geladen.')
else:
    print(f'Modell nicht"{model_file}" gefunden')
    exit()

y_combined_test = {'up_trend': y_up_test, 'down_trend': y_down_test}
results = model.evaluate(x_test, y_combined_test, verbose=7)

up_accuracy = results[3]
down_accuracy = results[4]

test_acc = (up_accuracy + down_accuracy)/2
print(f"Test accuracy: {test_acc}")
