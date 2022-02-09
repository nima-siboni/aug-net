import tensorflow as tf
import tensorflow.keras as keras


class AugNet(keras.Model):
    """
    A keras model with the additional functionality of returning the Jacobian of labels with respect to features."""
    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float64),))
    def return_jacobian(self, x):
        """
        Calculates and returns the derivative of the labels with respect to features at point x.
        :param x: the features vector at which the Jacobian should be calculated.
        :return: the Jacobian.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self(x, training=True)  # Forward pass
        res = tape.jacobian(y_pred, x)
        return res
