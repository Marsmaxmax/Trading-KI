import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout
from util.datasequencer import create_sequences
import os

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird


data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100, y_up, y_down = create_sequences(candles)

#******************
#******Modell******
#******************

input_close = Input(shape=(100,1))
input_open = Input(shape=(100,1))
input_high = Input(shape=(100,1))
input_low = Input(shape=(100,1))
input_rsi = Input(shape=(100,1))
input_ema20 = Input(shape=(100,1))
input_ema50 = Input(shape=(100,1))
input_ema100 = Input(shape=(100,1))

x = LSTM(64, return_sequences=False)(input_close)

output_up = Dense(1,name='up_trend')(x)
output_down = Dense(1,name='down_trend')(x)

# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_close, input_open, input_high, input_low, input_rsi, input_ema20, input_ema50,input_ema100], outputs=[output_up, output_down])

# Modell kompilieren
model.compile(optimizer='adam',loss={'up_trend': 'binary_crossentropy', 'down_trend': 'binary_crossentropy'}, metrics={'up_trend': 'accuracy', 'down_trend': 'accuracy'})

# Modellübersicht anzeigen
model.summary()

# Training des Modells
y_combined = {'up_trend': y_up, 'down_trend': y_down}

history = model.fit([x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100], y_combined, epochs=1, batch_size=32, validation_split=0.2)

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')

# # Vorhersagen treffen
# predictions_up, predictions_down = model.predict(X_test)

# # Beispielausgabe der Vorhersagen
# for i in range(5):
#     print(f"Aufwärtstrend Wahrscheinlichkeit: {predictions_up[i][0]:.2f}, Abwärtstrend Wahrscheinlichkeit: {predictions_down[i][1]:.2f}")
