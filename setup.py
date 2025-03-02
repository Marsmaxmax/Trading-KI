import os
import keras
import pandas as pd
import numpy as np
from keras.api.models import Model
from keras.api.regularizers import l2
from keras.api.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten
import tensorflow as tf
from util.custom.customCallback import PrintLR
from util.custom.customError import SemiLinearSquared
from preperation.datasequencer import create_sequences
from config import CHECKPOINT_DIR, INPUT_LENGTH, MODEL_FILE
input_file = 'data/XAUUSD/train_1.csv'  # Name der Eingabedatei



data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)

#******************
#******Modell******
#******************
# print(x_candle.shape)#Debug
# print(x_ema.shape)
# print(y_direction.shape)
# print(y_long.shape)
# print(y_short.shape)

input_candle = Input(shape=(INPUT_LENGTH,4))
input_ema = Input(shape=(INPUT_LENGTH,3))

candle = LSTM(64, return_sequences = True)(input_candle)
candle = LSTM(64, return_sequences = True)(candle)
candle = Dense(64,kernel_regularizer = l2(0.01))(candle)
candle = LSTM(64, return_sequences = True)(candle)


ema = LSTM(64, return_sequences = True)(input_ema)
ema = LSTM(64, return_sequences = True)(ema)
ema = Dense(64,kernel_regularizer = l2(0.01), activation='sigmoid')(ema)
ema = LSTM(64, return_sequences = True)(ema)



all = Dense(64,kernel_regularizer = l2(0.01))(Concatenate()([candle, ema]))

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


checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}.weights.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    # tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    PrintLR()
]
history = model.fit([x_candle, x_ema], [y_direction, y_long, y_short], epochs=1, batch_size=32, validation_split=0.2, callbacks=callbacks)

model.save(MODEL_FILE)
print(f'Modell wurde als "{MODEL_FILE}" gespeichert.')

