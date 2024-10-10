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
x_close, x_open, x_high, x_low, x_rsi, x_ema20, x_ema50, x_ema100, y_up, y_down = create_sequences(candles)

#******************
#******Modell******
#******************

input_close = Input(shape=(128,1))
input_open = Input(shape=(128,1))
input_high = Input(shape=(128,1))
input_low = Input(shape=(128,1))
input_rsi = Input(shape=(128,1))
input_ema20 = Input(shape=(128,1))
input_ema50 = Input(shape=(128,1))
input_ema100 = Input(shape=(128,1))

branch_0 = LSTM(128, return_sequences=True) (input_close)
branch_0 = LSTM(128, return_sequences=True) (branch_0)
branch_0 = Dense(128, activation = 'silu') (branch_0)
branch_0 = LSTM(128, return_sequences=True) (branch_0)
branch_0 = LSTM(128, return_sequences=True) (branch_0)

branch_1 = LSTM(128, return_sequences=True) (input_open)
branch_1 = LSTM(128, return_sequences=True) (branch_1)
branch_1 = Dense(128, activation = 'silu') (branch_1)
branch_1 = LSTM(128, return_sequences=True) (branch_1)
branch_1 = LSTM(128, return_sequences=True) (branch_1)

branch_2 = LSTM(128, return_sequences=True) (input_high)
branch_2 = LSTM(128, return_sequences=True) (branch_2)
branch_2 = Dense(128, activation = 'silu') (branch_2)
branch_2 = LSTM(128, return_sequences=True) (branch_2)
branch_2 = LSTM(128, return_sequences=True) (branch_2)

branch_3 = LSTM(128, return_sequences=True) (input_low)
branch_3 = LSTM(128, return_sequences=True) (branch_3)
branch_3 = Dense(128, activation = 'silu') (branch_3)
branch_3 = LSTM(128, return_sequences=True) (branch_3)
branch_3 = LSTM(128, return_sequences=True) (branch_3)

branch_4 = LSTM(128, return_sequences=True) (input_rsi)
branch_4 = LSTM(128, return_sequences=True) (branch_4)
branch_4 = Dense(128, activation = 'silu') (branch_4)
branch_4 = LSTM(128, return_sequences=True) (branch_4)
branch_4 = LSTM(128, return_sequences=True) (branch_4)

branch_5 = LSTM(128, return_sequences=True) (input_ema20)
branch_5 = LSTM(128, return_sequences=True) (branch_5)
branch_5 = Dense(128, activation = 'silu') (branch_5)
branch_5 = LSTM(128, return_sequences=True) (branch_5)
branch_5 = LSTM(128, return_sequences=True) (branch_5)

branch_6 = LSTM(128, return_sequences=True) (input_ema50)
branch_6 = LSTM(128, return_sequences=True) (branch_6)
branch_6 = Dense(128, activation = 'silu') (branch_6)
branch_6 = LSTM(128, return_sequences=True) (branch_6)
branch_6 = LSTM(128, return_sequences=True) (branch_6)

branch_7 = LSTM(128, return_sequences=True) (input_ema100)
branch_7 = LSTM(128, return_sequences=True) (branch_7)
branch_7 = Dense(128, activation = 'silu') (branch_7)
branch_7 = LSTM(128, return_sequences=True) (branch_7)
branch_7 = LSTM(128, return_sequences=True) (branch_7)

alltensors = Concatenate() ([branch_0,branch_1,branch_2,branch_3,branch_4,branch_5,branch_6,branch_7])
datatensors = Concatenate() ([branch_0,branch_1,branch_2,branch_3])
inditensors = Concatenate() ([branch_4,branch_5,branch_6,branch_7])

alltensors = LSTM(128, return_sequences=True) (alltensors)
alltensors = LSTM(128, return_sequences=True) (alltensors)
alltensors = Dense(128, activation = 'relu6') (alltensors)
alltensors = LSTM(128, return_sequences=True) (alltensors)
alltensors = LSTM(128, return_sequences=True) (alltensors)

branch_8  = LSTM(128, return_sequences=True) (datatensors)
branch_8  = LSTM(128, return_sequences=True) (branch_8)
branch_8 = Dense(128, activation = 'silu') (branch_8)
branch_8  = LSTM(128, return_sequences=True) (branch_8)
branch_8 = Concatenate() ([alltensors, branch_8])
branch_8  = LSTM(128, return_sequences=True) (branch_8)
branch_8 = Dense(128, activation = 'silu') (branch_8)
branch_8  = LSTM(128, return_sequences=True) (branch_8)
branch_8  = LSTM(128, return_sequences=True) (branch_8)

branch_9  = LSTM(128, return_sequences=True) (datatensors)
branch_9  = LSTM(128, return_sequences=True) (branch_9)
branch_9 = Dense(128, activation = 'silu') (branch_9)
branch_9  = LSTM(128, return_sequences=True) (branch_9)
branch_9 = Concatenate() ([alltensors, branch_9])
branch_9  = LSTM(128, return_sequences=True) (branch_9)
branch_9 = Dense(128, activation = 'silu') (branch_9)
branch_9  = LSTM(128, return_sequences=True) (branch_9)
branch_9  = LSTM(128, return_sequences=True) (branch_9)

branch_10  = LSTM(128, return_sequences=True) (inditensors)
branch_10  = LSTM(128, return_sequences=True) (branch_10)
branch_10 = Dense(128, activation = 'silu') (branch_10)
branch_10  = LSTM(128, return_sequences=True) (branch_10)
branch_10 = Concatenate() ([alltensors, branch_10])
branch_10  = LSTM(128, return_sequences=True) (branch_10)
branch_10 = Dense(128, activation = 'silu') (branch_10)
branch_10  = LSTM(128, return_sequences=True) (branch_10)
branch_10  = LSTM(128, return_sequences=True) (branch_10)

branch_11  = LSTM(128, return_sequences=True) (inditensors)
branch_11  = LSTM(128, return_sequences=True) (branch_11)
branch_11 = Dense(128, activation = 'silu') (branch_11)
branch_11  = LSTM(128, return_sequences=True) (branch_11)
branch_11 = Concatenate() ([alltensors, branch_11])
branch_11  = LSTM(128, return_sequences=True) (branch_11)
branch_11 = Dense(128, activation = 'silu') (branch_11)
branch_11  = LSTM(128, return_sequences=True) (branch_11)
branch_11  = LSTM(128, return_sequences=True) (branch_11)

branch_12 = Concatenate() ([branch_8, branch_10])
branch_12  = LSTM(128, return_sequences=True) (branch_12)
branch_12  = LSTM(128, return_sequences=True) (branch_12)
branch_12 = Dense(128, activation = 'silu') (branch_12)
branch_12  = LSTM(128, return_sequences=True) (branch_12)
branch_12  = LSTM(128, return_sequences=True) (branch_12)
branch_12 = Concatenate() ([alltensors, branch_12])
branch_12  = LSTM(128, return_sequences=True) (branch_12)
branch_12  = LSTM(128, return_sequences=True) (branch_12)

branch_13 = Concatenate() ([branch_9, branch_11])
branch_13  = LSTM(128, return_sequences=True) (branch_13)
branch_13  = LSTM(128, return_sequences=True) (branch_13)
branch_13 = Dense(128, activation = 'silu') (branch_13)
branch_13  = LSTM(128, return_sequences=True) (branch_13)
branch_13  = LSTM(128, return_sequences=True) (branch_13)
branch_13 = Concatenate() ([alltensors, branch_13])
branch_13  = LSTM(128, return_sequences=True) (branch_13)
branch_13  = LSTM(128, return_sequences=True) (branch_13)

output_up = GlobalAveragePooling1D()(branch_12)
output_up = Dense(128)(output_up)
output_up = Dense(1,activation = 'linear',name='up_trend')(output_up)

output_down = GlobalAveragePooling1D()(branch_13)
output_down = Dense(128) (output_down)
output_down = Dense(1,activation = 'linear',name='down_trend')(output_down)

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
