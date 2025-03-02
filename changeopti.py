import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from config import INPUT_LENGTH, MODEL_FILE
from util.custom.customError import SemiLinearSquared
from util.custom.customfunctions import load_custom_model
from preperation.datasequencer import create_sequences

input_file = 'data/train1.csv'  # Name der Eingabedatei
data = pd.read_csv(input_file, header=None)
candles = data.values  # Close, Open, High, Low
x_candle, x_ema, y_direction, y_long, y_short = create_sequences(candles, INPUT_LENGTH)
# Überprüfen, ob das Modell existiert und geladen werden kann
model = load_custom_model(MODEL_FILE)

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

model.compile(optimizer=customoptimizer, loss=SemiLinearSquared(threshold=0.1, threshold_is_relative=True, regularization_factor=0.01), metrics=['mse','mae','accuracy'])
model.save(MODEL_FILE)