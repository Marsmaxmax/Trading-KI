import keras
import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable(package="Custom", name="SemiLinearSquared")
class SemiLinearSquared(keras.losses.Loss):
    '''Basicly a reverse Huber loss: Below and at thershold: linear, above squared
    '''
    def __init__(self, threshold = 1, regularization_factor=0.0, alpha=2, name='SemiLinearSquared'):
        super().__init__(name=name)
        self.threshold = threshold
        self.regularization_factor = regularization_factor
        self.alpha = alpha
    def call(self, y_true, y_pred):
        regularization = self.regularization_factor * tf.reduce_mean(tf.square(y_pred))
        abs_loss = tf.abs(y_pred - y_true)
        squ_loss = tf.square(y_pred - y_true)
        
        loss1 = abs_loss * self.alpha
        
        loss2 = ((squ_loss * self.alpha)/2*self.threshold)+((2*self.alpha*self.threshold)/4)

        loss = tf.where(abs_loss <= self.threshold, loss1, loss2)
        return loss + regularization
    def get_config(self):
        return {'name': self.name,
                'threshold': self.threshold,
                'regularization_factor': self.regularization_factor,
                'alpha': self.alpha}
    
@register_keras_serializable(package="Custom", name="NonZeroBCELoss")
class NonZeroBCELoss(keras.losses.Loss):
    """A BCE(Biniary Cross Entropy) Loss that punishes with input alpha, if y_pred is 0, or close to 0, but y_true is 1.
    Asymetrical loss fuction to use when positive cases are more important and/or more rare than negative cases.
    """
    def __init__(self,alpha=5.0, name='NonZeroBCELoss'):
        self.alpha = alpha
        super().__init__(name=name)
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        non_zero_penalty = self.alpha * tf.where((y_true == 1) & (y_pred < 0.25), 1.0, 0.0)*(2*y_pred)
        return bce + non_zero_penalty
    def get_config(self):
        return {'name': self.name,
                'alpha': self.alpha}
