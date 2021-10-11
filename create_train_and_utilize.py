"""This is a demonstration for AugNet."""
import copy
import numpy as np
import tensorflow as tf
from utils import create_the_model, create_a_dataset

# In this script we:
# 1 - Create a dummy dataset of size (None, 2, 4); the implementation is not important.
# 2 - Create an instance of AugNet
# 3 - Train the model
# 4 - Test the Jacobian, by obtaining it from Augnet and calculating it approximately by finite difference.

print("1: creating a dummy dataset ...")
x_training, y_training = create_a_dataset(nr_samples=1000, random_seed=1)

print("2: creating an instance of augnet model for training ...")
input_size = 4
output_size = 2
model = create_the_model(input_shape=input_size, output_shape=output_size, units=[128, 64, 32, 16], learning_rate=0.0001)

print("3: training started ...")
model.fit(x=x_training, y=y_training, epochs=5_000)
print("training finished.")

# 4 - getting the jacobian values for a test sample and compare it with the numerically (finite difference) values.
print(" 4: calculating the Jacobian")
# make sure it is a float64
test_sample = np.array([0., 0.0, 0.0, 0.0], dtype=np.float64)

# 4.1 - first obtaining the jacobian from AugNet; which uses automatic differentiation within tensorflow package.
test_value_tf = tf.Variable([test_sample])
jacobian_ad_estimate = model.return_the_input_grad(test_value_tf)


# 4.2 - calculate the jacobian via finite difference.
eps = 10 ** -5

for i in range(input_size):
    test_value_plus = copy.deepcopy(test_sample)
    test_value_plus[i] = test_value_plus[i] + eps
    test_value_minus = copy.deepcopy(test_sample)
    test_value_minus[i] = test_value_minus[i] - eps

    jacobian_fd_estimate = (model.predict(np.array([test_value_plus])) - model.predict(np.array([test_value_minus]))) /\
                         (2 * eps)

    print("Derivative of the elements of the output, y, with respect to x[", i, "], \n... numerically estimated derivative: ",
          jacobian_fd_estimate, "\n... Automatic Differentiation estimate: ", jacobian_ad_estimate[0, :, 0, i].numpy(), "\n")
