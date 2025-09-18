import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, GlobalAveragePooling1D
from config import INPUT_LENGTH, CHECKPOINT_DIR, MODEL_FILE
from util.custom.customCallback import PrintLR
from util.custom.customfunctions import load_custom_model
from preperation.datasequencer import create_sequences
import os
import sys


input_file = 'data/NASDAQ100_5m/output.csv'  # Name der Eingabedatei
training_set_1 = 'data/BTCUSDT_1m/2.5k_packs/pack_1.csv'
training_set_2 = 'data/BTCUSDT_1m/2.5k_packs/pack_2.csv'
training_set_3 = 'data/BTCUSDT_1m/2.5k_packs/pack_3.csv'
training_set_4 = 'data/BTCUSDT_1m/25k_packs/pack_72.csv'
training_sets = [training_set_1, training_set_2, training_set_3, training_set_4]
batch = 16
runs = int()

tf.debugging.set_log_device_placement(False)
# sys.argv[0] ist der Name des Skripts
# sys.argv[1] ist das erste Argument von der Kommandozeile
if len(sys.argv) > 1:
    print(f" Anzahl Durchläufe {sys.argv[1]}")
    print(f" Batch Größe{batch}")
    runs = int(sys.argv[1])
# elif len(sys.argv) > 2:
#     print(f" Batch Größe{sys.argv[2]}")
#     batch = int(sys.argv[2])
else:
    print("Keine Argumente eingegeben.")
    exit()

data = pd.read_csv(input_file, header=None)

candles = data.values  # Close, Open, High, Low

# Daten in Sequenzen umwandeln
X, Y = create_sequences(candles, INPUT_LENGTH)

model = load_custom_model(MODEL_FILE)
checkpoint_prefix = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}.weights.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    # tf.keras.callbacks.LearningRateScheduler(lr_schedule),
    PrintLR()
]
history = model.fit([X], [Y], epochs=runs, batch_size=batch, validation_split=0.2, callbacks=callbacks)

model.save(MODEL_FILE)
# model.summary()
print(f'Modell wurde als "{MODEL_FILE}" gespeichert.')