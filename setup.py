import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten
from util.datasequencer import create_sequences
import os

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird


data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_close, y_open, y_high, y_low = create_sequences(candles, 64)

#******************
#******Modell******
#******************

input_close = Input(shape=(64,1))
input_open = Input(shape=(64,1))
input_high = Input(shape=(64,1))
input_low = Input(shape=(64,1))
input_ema1 = Input(shape=(64,1))
input_ema2 = Input(shape=(64,1))
input_ema3 = Input(shape=(64,1))

close = LSTM(64, return_sequences = True)(Dense(128)(input_close))
open = LSTM(64, return_sequences = True)(Dense(128)(input_open))
high = LSTM(64, return_sequences = True)(Dense(128)(input_high))
low = LSTM(64, return_sequences = True)(Dense(128)(input_low))
ema1 = LSTM(64, return_sequences = True)(Dense(128)(input_ema1))
ema2 = LSTM(64, return_sequences = True)(Dense(128)(input_ema2))
ema3 = LSTM(64, return_sequences = True)(Dense(128)(input_ema3))

copl = LSTM(64, return_sequences = True)(LSTM(64, return_sequences = True)(Concatenate()([close, open, high, low])))
emas = LSTM(64, return_sequences = True)(LSTM(64, return_sequences = True)(Concatenate()([ema1, ema2, ema3])))
all = Dense(128, activation ='sigmoid')(LSTM(64, return_sequences = True)(LSTM(64, return_sequences = True)(Concatenate()([copl, emas]))))

close = LSTM(64, return_sequences = False)(LSTM(64, return_sequences = True)(Concatenate()([all, close])))
open = LSTM(64, return_sequences = False)(LSTM(64, return_sequences = True)(Concatenate()([all, open])))
high = LSTM(64, return_sequences = False)(LSTM(64, return_sequences = True)(Concatenate()([all, high])))
low = LSTM(64, return_sequences = False)(LSTM(64, return_sequences = True)(Concatenate()([all, low])))

output_close = Dense(1,activation = 'linear',name='output_close')(Flatten()(close))
output_open = Dense(1,activation = 'linear',name='output_open')(Flatten()(open))
output_high = Dense(1,activation = 'linear',name='output_high')(Flatten()(high))
output_low = Dense(1,activation = 'linear',name='output_low')(Flatten()(low))


# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_close, input_open, input_high, input_low, input_ema1, input_ema2,input_ema3], outputs=[output_close,output_open,output_high,output_low])

# Modell kompilieren
model.compile(optimizer='adam',loss={'output_close': 'binary_crossentropy', 'output_open': 'binary_crossentropy', 'output_high': 'binary_crossentropy', 'output_low': 'binary_crossentropy'}, metrics={'output_close': 'accuracy', 'output_open': 'accuracy', 'output_high': 'accuracy', 'output_low': 'accuracy'})

# Modellübersicht anzeigen
model.summary()
# Training des Modells
history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], [y_close, y_open, y_high, y_low], epochs=1, batch_size=32, validation_split=0.2)

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')