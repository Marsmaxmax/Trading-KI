import keras
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="ProfitLoss")
class Profitloss(keras.losses.Loss):
    def __init__(self, regularization_factor=0.0, name='Profitloss'):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor
    def call(self, y_true, y_pred):
        regularization = self.regularization_factor * tf.reduce_mean(tf.square(y_pred))
        profit_loss = -tf.math.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return profit_loss + regularization
    def get_config(self):
        return {'name': self.name, 'regularization_factor': self.regularization_factor}
