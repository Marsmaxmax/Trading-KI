import os
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D, Flatten, BatchNormalization, Reshape, GlobalMaxPooling1D
from util.custom.customCallback import PrintLR
from util.custom.customLoss import NonZeroBCELoss, SemiLinearSquared
from keras.losses import BinaryCrossentropy
from preperation.datasequencer import create_sequences
from config import CHECKPOINT_DIR, INPUT_LENGTH, MODEL_FILE
from util.layerblocks import TransformerStack, DenseStack, LSTMStack
input_file = 'data/BTCUSDT_1m/10k_packs/pack_1.csv'  # Name der Eingabedatei



data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
X, Y = create_sequences(candles, INPUT_LENGTH)



#******************
#******Modell******
#******************
#Inputs:Candles, Open Position, Account Balance
#
#
x = Input(shape=(INPUT_LENGTH,4), name='input')
y = Dense(256)(x) 
y = TransformerStack(y, num_blocks=4, embed_dim=64, num_heads=4, ff_dim=128, rate=0.1)
y = GlobalAveragePooling1D()(y)
y = Dense(256)(y)
y = Dense(64)(y)
y = Dense(16) (y)
y = Dense(4, name='output')(y)

# Modell mit mehreren Ausgaben erstellen
model = Model(inputs=[x], outputs=[y])
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=False
)

customoptimizer = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    # rho=0.9,           
    # momentum=0.8,      
    epsilon=1e-7,      
    # centered=False,      
    clipnorm=1.0,       
    clipvalue=None,       
    global_clipnorm=None 
    )

model.compile(optimizer=customoptimizer, 
              loss=SemiLinearSquared(0.1,0, 5,), 
              metrics=['accuracy'])

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
history = model.fit([X], [Y], epochs=1, batch_size=32, validation_split=0.2, callbacks=callbacks)

model.save(MODEL_FILE)
print(f'Modell wurde als "{MODEL_FILE}" gespeichert.')

