"""This is a demonstration for AugNet."""
import copy
import numpy as np
import tensorflow as tf
from utils import create_the_model, create_a_dataset, calculate_jacobian_with_finite_difference

# In this script we:
# 1 - Create a dummy dataset of size (None, 2, 4); the implementation is not important.
# 2 - Create an instance of AugNet
# 3 - Train the model
# 4 - Test the Jacobian, by obtaining it from Augnet and calculating it approximately by finite difference.

print("1: creating a dummy dataset ...")
x_training, y_training = create_a_dataset(nr_samples=1000, random_seed=1)

print("2: creating an instance of AugNet model for training ...")
input_size = 4
output_size = 2
model = create_the_model(input_shape=input_size, output_shape=output_size, units=[64, 32, 16], learning_rate=0.0001)

print("3: training started ...")
model.fit(x=x_training, y=y_training, epochs=1000)
print("training finished.")

# 4 - getting the jacobian values for a test sample and compare it with the numerically (finite difference) values.
print(" 4: calculating the Jacobian")
# make sure it is a float64
test_sample = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

# 4.1 - first obtaining the jacobian from AugNet; which uses automatic differentiation within tensorflow package.
test_sample_tf = tf.Variable([test_sample])
jacobian_ad_estimate = model.return_jacobian(test_sample_tf)


# 4.2 - calculate the jacobian via finite difference.
eps = 10 ** -5
jacobian_fd_estimate = calculate_jacobian_with_finite_difference(test_sample, eps=eps, model=model)

# Compare the AD, and FD approximation
for i in range(input_size):
    print("Derivative of the elements of the output, y, with respect to x[", i, "],",
          "\n... numerically estimated derivative: ", jacobian_fd_estimate[i],
          "\n... Automatic Differentiation estimate: ", jacobian_ad_estimate[0, :, 0, i].numpy(), "\n")
