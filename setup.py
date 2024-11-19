import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from util.datasequencer import create_sequences
import os

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird


data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_ema20, x_ema50, x_ema100, y_close, y_open, y_high, y_low = create_sequences(candles, 64)

#******************
#******Modell******
#******************

input_close = Input(shape=(64,1))
input_open = Input(shape=(64,1))
input_high = Input(shape=(64,1))
input_low = Input(shape=(64,1))
input_ema20 = Input(shape=(64,1))
input_ema50 = Input(shape=(64,1))
input_ema100 = Input(shape=(64,1))



output_close = Dense(1,activation = 'linear',name='output_close')(input_close)
output_open = Dense(1,activation = 'linear',name='output_open')(input_open)
output_high = Dense(1,activation = 'linear',name='output_high')(input_high)
output_low = Dense(1,activation = 'linear',name='output_low')(input_low)


# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_close, input_open, input_high, input_low, input_ema20, input_ema50,input_ema100], outputs=[output_close,output_open,output_high,output_low])

# Modell kompilieren
model.compile(optimizer='adam',loss={'output_close': 'binary_crossentropy', 'output_open': 'binary_crossentropy', 'output_high': 'binary_crossentropy', 'output_low': 'binary_crossentropy'}, metrics={'output_close': 'accuracy', 'output_open': 'accuracy', 'output_high': 'accuracy', 'output_low': 'accuracy'})

# Modellübersicht anzeigen
model.summary()
print(y_close.shape)
print(y_open.shape)
print(x_high.shape)
print(x_low.shape)
# Training des Modells
history = model.fit([x_close, x_open, x_high, x_low, x_ema20, x_ema50, x_ema100], [y_close, y_open, y_high, y_low], epochs=1, batch_size=32, validation_split=0.2)

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')

# # Vorhersagen treffen
# predictions_up, predictions_down = model.predict(X_test)

# # Beispielausgabe der Vorhersagen
# for i in range(5):
#     print(f"Aufwärtstrend Wahrscheinlichkeit: {predictions_up[i][0]:.2f}, Abwärtstrend Wahrscheinlichkeit: {predictions_down[i][1]:.2f}")
