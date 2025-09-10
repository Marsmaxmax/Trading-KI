import os
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten, BatchNormalization, Reshape
from util.custom.customCallback import PrintLR
from util.custom.customLoss import SemiLinearSquared
from keras.losses import BinaryCrossentropy
from preperation.datasequencer import create_sequences
from config import CHECKPOINT_DIR, INPUT_LENGTH, MODEL_FILE
from util.layerblocks import TransformerStack, DenseStack, LSTMStack
input_file = 'data/BTCUSDT_1m/10k_packs/pack_1.csv'  # Name der Eingabedatei



data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
X_candles, X_balance, X_position, Y_long, Y_short, Y_hold, Y_close = create_sequences(candles, INPUT_LENGTH)
print(X_candles.shape, X_balance.shape, X_position.shape, Y_long.shape, Y_short.shape, Y_hold.shape, Y_close.shape)
#******************
#******Modell******
#******************
#Inputs:Candles, Open Position, Account Balance
#
#
input_candle = Input(shape=(INPUT_LENGTH, 4), name='candle_input')
input_balance = Input(shape=(1,), name='balance_input')
input_position = Input(shape=(2,), name='position_input')



position_analysis = Dense(4*INPUT_LENGTH, activation='sigmoid')(input_position)
position_analysis = Reshape((INPUT_LENGTH, 4))(position_analysis)

balance_analysis = Dense(4*INPUT_LENGTH, activation='sigmoid')(input_balance)
balance_analysis = Reshape((INPUT_LENGTH, 4))(balance_analysis)


position_vs_candles= Concatenate()([position_analysis,input_candle])
position_vs_candles = DenseStack(position_vs_candles, num_blocks=1, embed_dim=32)

balance_vs_candles = Concatenate()([balance_analysis,input_candle])
balance_vs_candles = DenseStack(balance_vs_candles, num_blocks=1, embed_dim=32)

bal_pos_vs_candles = Concatenate()([position_vs_candles,balance_vs_candles])
bal_pos_vs_candles = DenseStack(bal_pos_vs_candles, num_blocks=1, embed_dim=32)


# Transformer-Pfad
x1 = TransformerStack(input_candle, num_blocks=2, embed_dim=64, num_heads=2, ff_dim=128)
x1 = DenseStack(x1, num_blocks=1, embed_dim=128)
x1 = DenseStack(x1, num_blocks=1, embed_dim=64)

# LSTM-Pfad
x2 = LSTMStack(input_candle, num_blocks=1, units=64)
x2 = DenseStack(x2, num_blocks=1, embed_dim=128)
x2 = DenseStack(x2, num_blocks=1, embed_dim=64)

# Zusammenführen
trafo_LSTM = Concatenate()([x1, x2])
trafo_LSTM = DenseStack(trafo_LSTM, num_blocks=1, embed_dim=128)
all = Concatenate()([trafo_LSTM, bal_pos_vs_candles])
all = TransformerStack(all, num_blocks=1, embed_dim=64, num_heads=2, ff_dim=128)
all = LSTM(128, return_sequences=True)(all)
all = LSTM(128, return_sequences=False)(all)

#Short
short = DenseStack(all, num_blocks=1, embed_dim=32)
yshort = Dense(1, activation='sigmoid', name='short')(short)
#Long
long = DenseStack(all, num_blocks=1, embed_dim=32)
ylong = Dense(1, activation='sigmoid', name='long')(long)
#Hold
hold = DenseStack(all, num_blocks=1, embed_dim=32)
yhold = Dense(1, activation='sigmoid', name='hold')(hold)
#Close Position
close = DenseStack(all, num_blocks=1, embed_dim=32)
yclose = Dense(1, activation='sigmoid', name='close')(close)

# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[input_candle, input_position, input_balance], outputs=[yshort,ylong,yhold,yclose])
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

model.compile(optimizer=customoptimizer, loss=BinaryCrossentropy(), metrics=['accuracy','accuracy', 'accuracy','accuracy'])

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
history = model.fit([X_candles, X_position, X_balance], [Y_long, Y_short, Y_hold, Y_close], epochs=1, batch_size=32, validation_split=0.2, callbacks=callbacks)

model.save(MODEL_FILE)
print(f'Modell wurde als "{MODEL_FILE}" gespeichert.')

