import keras
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="SemiLinearSquared")
class SemiLinearSquared(keras.losses.Loss):
    def __init__(self, threshold = 0.1, threshold_is_relative = True, regularization_factor=0.0, name='SemiLinearSquared'):
        super().__init__(name=name)
        self.threshold = threshold
        self.threshold_is_relative = threshold_is_relative
        self.regularization_factor = regularization_factor
    def call(self, y_true, y_pred):
        regularization = self.regularization_factor * tf.reduce_mean(tf.square(y_pred))
        error = y_true-y_pred
        square = tf.square(error)
        linear = tf.abs(error)
        if self.threshold_is_relative:
            threshold = self.threshold * tf.abs(y_true)
        else:
            threshold = self.threshold
        
        loss = tf.where(tf.abs(error) <= threshold, linear, square)
        return loss + regularization
    def get_config(self):
        return {'name': self.name,
                'threshold': self.threshold,
                'threshold_is_relative': self.threshold_is_relative,
                'regularization_factor': self.regularization_factor}

@register_keras_serializable(package="Custom", name="ProfitOrientedLoss")   
class ProfitOrientedLoss(keras.losses.Loss):
    def __init__(self, name="ProfitOrientedLoss", dtype=None):
        super().__init__(name=name)
    
