import tensorflow as tf

from config import MODEL_FILE
from util.custom.customfunctions import load_custom_model


class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    model = load_custom_model(MODEL_FILE)
    print('\nLearning rate for epoch {} is {}'.format(        epoch + 1, model.optimizer.learning_rate.numpy()))