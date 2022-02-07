""""Some utilities which are not AugNet specific."""
import copy
import sys

import numpy as np
import tensorflow.keras as keras
from typing import List, Tuple
from aug_model import AugNet


def create_the_model(input_shape: int = 4, output_shape: int = 2, units: List = [16, 8], learning_rate: float = 0.0001)\
        -> AugNet:
    """
    creates and returns am instance of AugNet, i.e. a model which can return the Jacobian of the labels with respect to
    features.

    :param learning_rate: the learning rate
    :param input_shape: the length of the input vector, an integer is expected.
    :param output_shape: the length of the output vector, an integer is expected.
    :param units: a list of number of neurons in the hidden layers, e.g. [16, 8] creates two hidden layers with 16 and
    8 neurons respectively.
    :return: am instance of AugNet.
    """
    inputs = keras.layers.Input(input_shape)
    x = inputs
    for counter, nr_neurons in enumerate(units):
        x = keras.layers.Dense(units=nr_neurons, activation='sigmoid', name='dense_layer' + str(counter))(x)

    outputs = keras.layers.Dense(units=output_shape, activation='linear')(x)

    model = AugNet(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='MSE')

    return model


def create_a_dataset(nr_samples: int = 1000, random_seed: int = 1) -> Tuple:
    """
    Creates a dataset consists of samples with 4 features and 2 labels.

    .. note::
    features are x = [x1, x2, x3, x4], and labels are y = [sin(x1 * x2), cos(x3 * x4)] , and x are randomly chosen in
    [-1, 1]^4 hypercube.

    :param nr_samples: number of samples to be returned.
    :param random_seed: the random seed.

    :returns: x, y
    """
    np.random.seed(random_seed)
    x = np.random.rand(nr_samples, 4)
    y = np.array([np.sin(x[:, 0] * x[:, 1]), np.cos(x[:, 2] * x[:, 3])])
    y = np.reshape(y, (-1, 2))
    return x, y


def calculate_jacobian_with_finite_difference(x, eps, model):
    """
    Returns Jacobian at x for model
    :param x: the Jacobian is calculated at this point
    :param eps: the infinitesimal value for calculation of Jacobian via finite difference.
    :param model: the model for outputs of which the Jac. is calculated.
    :return: Jacobian.
    """
    jacobian_fd_estimate = []

    for i in range(len(x)):
        test_value_plus = copy.deepcopy(x)
        test_value_plus[i] = test_value_plus[i] + eps
        test_value_minus = copy.deepcopy(x)
        test_value_minus[i] = test_value_minus[i] - eps

        tmp = (model.predict(np.array([test_value_plus])) - model.predict(np.array([test_value_minus]))) / (2 * eps)
        jacobian_fd_estimate.append(tmp)

    return jacobian_fd_estimate


def check_similarities(model_1, model_2):
    """
    Checks similarities of two models.
    :param model_1: the first keras model
    :param model_2: the second keras model.
    :return: True if the models are similar, otherwise breaks!
    """
    # Checking whether the parameters of the networks are same.
    residual = 0.0
    if len(model_1.weights) != len(model_2.weights):
        sys.exit("Error: Models do not have same structure!")
    for weight_id in range(len(model_1.weights)):
        residual += (np.max(np.abs(model_1.weights[weight_id] - model_2.weights[weight_id])))
    if residual != 0:
        sys.exit("Error: Models do not have the same values for parameters!")

    return True
