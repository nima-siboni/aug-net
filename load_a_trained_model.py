import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tensorflow import Variable
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import load_img
from aug_model import AugNet
from utils import check_similarities

# 0. Config
trained_model_dir = "trained_model"  # directory which has the trained model; could be an instance of AugNet or not.
sample_input = 'sample_input/sample_input.jpg'
jacobian_saving_dir = 'jacobian_results'
# 1. Load a trained model and create a similar AugNet
trained_model = keras.models.load_model(trained_model_dir)
model = AugNet(inputs=trained_model.inputs, outputs=trained_model.outputs)
if check_similarities(trained_model, model):
    print("The AugNet model is created successfully from ", trained_model_dir)
del trained_model

# 2. Getting the jacobian
# 2.1. read an input
x = np.array(load_img(sample_input, color_mode='grayscale', target_size=(256, 256))) / 255.
print("the shape of the input image is", x.shape)
# 2.2. Convert it to a tf variable
x_tf = Variable([x])
# 2.3. Calculating the predictions and/or the Jacobian (plus timing)
# 2.3.1 Getting the prediction
start = time.time()
prediction = np.squeeze(model.predict(np.array([x])))
end = time.time()
print("the prediction for the given input is\n", prediction)
print("the prediction is calculated in ", end - start, "sec.")
# 2.3.2 Calculate the Jacobian
start = time.time()
jacobian = np.squeeze(model.return_jacobian(x_tf))
end = time.time()
print("the jacobian for the input is also calculated and has the shape:", jacobian.shape,
      "in ", end - start, "sec.")

# 3. Saving the Jacobian
print("The results are saved in ", jacobian_saving_dir)
os.makedirs(jacobian_saving_dir, exist_ok=True)
for c, j in enumerate(jacobian):
    plt.imshow(j)
    plt.title("d prediction[" + str(c) + "] / d input")
    plt.savefig(os.path.join(jacobian_saving_dir, "d_pred_" + str(c) + "d_image.png"))
print("Good-bye")
