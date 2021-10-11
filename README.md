# aug-net
A small tutorial on how to calculate the Jacobian of the outputs wrt the inputs


# Introduction

Consider the case that you have a regression or a classification problem for which you have trained a neural network. Let's say the you have chosen the network well and it learns to predict the results correctly for training set! Is that enough, of course not! You have check the performance on the dev set, to make sure that you are not overfitting. You can go one step further in checking the performance of the network by using a test set, which is used only after all the experiments you did with your network.

Is that it? Can I know more about the performance of my model? Can I know what my model has learned?

Answers to these questions, partially, fall into explainable AI (XAI) where one attempts at figuring out the quality of an AI solution. One starting point to this analysis, can be asking the following questions:

"what is the sensitivity of my predictions to different elements of the input?". For example, let's say I have a cat/dog classifier, and it predicts that in the current image there is a dog. The above question translates to "changing which pixels has the most dramatic effect on the prediction and which picxels are not that important?". As you can see, this is going to give us some understanding of what our network has learned!


The sensitivity mention there has a technical term in mathematics; it is referred to as the Jacobian. Here, we augement a common keras DNN model with a method which return the Jacobian.

# Requirements
You can get the requirements by 

```
pip install -r requirements.txt
```

# How to use it?

In the file, ```create_train_and_utilize.py```, a neural network is created, trained, and then its Jacobian is calculated for a test input. In general:
* You have to first create a neural network as you usually do in Keras, but this time you use ```AugNet``` instead of ```keras.Model``` for compilin your model. ```AugNet``` is our subclass of keras.Model.
* Training is done via ```fit``` method of Keras as before.
* Access the Jacobian via ```return_jacobian``` method, e.g. ```my_model.return_jacobian()```
