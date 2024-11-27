import keras
import pandas as pd
import numpy as np
from keras.api.models import Model
from keras.api.regularizers import l2
from keras.api.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten
import tensorflow as tf
from util.customError import SemiLinearSquared
from util.datasequencer import create_sequences

input_file = 'data/train1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird


data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3, y_candle = create_sequences(candles, 64)

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

close = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_close))
open = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_open))
high = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_high))
low = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_low))
ema1 = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_ema1))
ema2 = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_ema2))
ema3 = Dense(64)(Dense(128,kernel_regularizer = l2(0.01))(input_ema3))

cohl = LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(Dense(128)(Concatenate()([close, open, high, low]))))
all = Dense(128, activation ='linear')(LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(LSTM(64, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(Dense(128,activation = 'relu',kernel_regularizer = l2(0.01))(Concatenate()([cohl, ema1,ema2,ema3])))))))
all = Dense(64,kernel_regularizer = l2(0.01))(LSTM(64, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1)(LSTM(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1)(all)))


output = Dense(4, activation='linear', name='output')(all)

print(y_candle.shape)
# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_close, input_open, input_high, input_low, input_ema1, input_ema2,input_ema3], outputs=output)

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.1,
    staircase=True
)

customoptimizer = keras.optimizers.RMSprop(
    learning_rate=lr_schedule,
    rho=0.9,           
    momentum=0.8,      
    epsilon=1e-7,      
    centered=True,      
    clipnorm=1.0,       
    clipvalue=None,       
    global_clipnorm=None 
    )

model.compile(optimizer=customoptimizer, loss=SemiLinearSquared(0.1), metrics=['mse','mae','accuracy'])

# Modellübersicht anzeigen
model.summary()
# Training des Modells
history = model.fit([x_close, x_open, x_high, x_low, x_ema1, x_ema2, x_ema3], y_candle, epochs=1, batch_size=32, validation_split=0.2)

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')