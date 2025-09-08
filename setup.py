import os
import keras
import pandas as pd
import numpy as np
from keras.api.models import Model
from keras.api.regularizers import l2
from keras.api.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten, BatchNormalization
import tensorflow as tf
from util.custom.customCallback import PrintLR
from util.custom.customError import SemiLinearSquared
from keras.api.losses import BinaryCrossentropy
from preperation.datasequencer import create_sequences
from config import CHECKPOINT_DIR, INPUT_LENGTH, MODEL_FILE
from util.layerblocks import TransformerStack, DenseStack, LSTMStack
input_file = 'data/XAUUSD/train_1.csv'  # Name der Eingabedatei



data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
X, y_long, y_short = create_sequences(candles, INPUT_LENGTH)

#******************
#******Modell******
#******************


input_candle = Input(shape=(INPUT_LENGTH, 4))

# Transformer-Pfad
x1 = TransformerStack(input_candle, num_blocks=4, embed_dim=64, num_heads=2, ff_dim=128)
x1 = DenseStack(x1, num_blocks=4, embed_dim=64)

# LSTM-Pfad
x2 = LSTMStack(input_candle, num_blocks=4, units=64)
x2 = DenseStack(x2, num_blocks=4, embed_dim=64)

# Zusammenführen
all = Concatenate()([x1, x2])
all = DenseStack(all, num_blocks=4, embed_dim=64)
all = TransformerStack(all, num_blocks=8, embed_dim=64, num_heads=2, ff_dim=128)
all = LSTMStack(all, num_blocks=8
                , units=64)
# Long-Ausgabe
long_lstm = LSTM(64, return_sequences=False)(all)
long_dense = Dense(64, activation='sigmoid')(long_lstm)
long_dense = Dense(16, activation='sigmoid')(long_dense)
output_long = Dense(1, activation='sigmoid')(long_dense)

# Short-Ausgabe
short_lstm = LSTM(64, return_sequences=False)(all)
short_dense = Dense(64, activation='sigmoid')(short_lstm)
short_dense = Dense(16, activation='sigmoid')(short_dense)
output_short = Dense(1, activation='sigmoid')(short_dense)
# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_candle], outputs=[output_long, output_short])
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.1,
    staircase=False
)

customoptimizer = keras.optimizers.RMSprop(
    learning_rate=lr_schedule,
    rho=0.9,           
    momentum=0.8,      
    epsilon=1e-7,      
    centered=False,      
    clipnorm=1.0,       
    clipvalue=None,       
    global_clipnorm=None 
    )

model.compile(optimizer=customoptimizer, loss=BinaryCrossentropy(), metrics=['accuracy','accuracy'])

# Modellübersicht anzeigen
model.summary()
# Training des Modells


checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}.weights.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True),
    # tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    # PrintLR()
]
history = model.fit([X], [y_long, y_short], epochs=1, batch_size=32, validation_split=0.2, callbacks=callbacks)

model.save(MODEL_FILE)
print(f'Modell wurde als "{MODEL_FILE}" gespeichert.')

