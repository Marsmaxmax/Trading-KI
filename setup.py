import keras
import pandas as pd
import numpy as np
from keras.api.models import Model
from keras.api.regularizers import l2
from keras.api.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten
import tensorflow as tf
from util.custom.customError import SemiLinearSquared
from preperation.datasequencer import create_sequences
from config import INPUT_LENGTH
input_file = 'data/XAUUSD/train_1.csv'  # Name der Eingabedatei
model_file = 'trend_model.keras'  # Name der Datei, in der das Modell gespeichert wird


data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)

#******************
#******Modell******
#******************
print(x_candle.shape)
print(x_ema.shape)
print(y_direction.shape)
print(y_long.shape)
print(y_short.shape)

input_candle = Input(shape=(INPUT_LENGTH,4))
input_ema = Input(shape=(INPUT_LENGTH,3))



all = Dense(64,kernel_regularizer = l2(0.01))(Concatenate()([input_candle, input_ema]))

output_direction = Dense(1, activation='linear')(GlobalAveragePooling1D()(all))
output_long = Dense(3, activation='linear')(GlobalAveragePooling1D()(all))
output_short = Dense(3, activation='linear')(GlobalAveragePooling1D()(all))

# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_candle, input_ema], outputs=[output_direction, output_long, output_short])

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
history = model.fit([x_candle, x_ema], [y_direction, y_long, y_short], epochs=1, batch_size=32, validation_split=0.2)

# Modell speichern
model.save(model_file)
print(f'Modell wurde als "{model_file}" gespeichert.')