
# coding: utf-8
# Car Detection | Cifar10 | GPU Based | CNN
# by Wyv3rn

import os
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPooling2D, Flatten, Dropout
import numpy as np 
# Using GPU :
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Load Train Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.


# Create Model :
model = Sequential()

# 32 Neurons 3,3 filter
# input_shape 32x32 pixel RGB

# Input Layer
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32,32,3), activation="relu", padding="same"))

# Hidden Layer 1
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"))

# Filter smoothing
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Hidden Layer 3
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))

# Hidden Layer 4
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Hidden Layer 5
model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))

# Hidden Layer 6
model.add(Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Hidden Layer 7
model.add(Flatten())
model.add(Dense(128, activation="relu"))

#output
model.add(Dense(1, activation="sigmoid"))

# 1 Input , 4 Hidden , 1 Output Layer

# by One Output only use := binary_crossentropy
model.compile(optimizer="rmsprop", loss="binary_crossentropy" , metrics=["accuracy"])

# Train Model with Parameter y_train and if y_train = 1 sein say True
y_train_car = y_train == 1

# Shuffle=True | mix data for better learning
model.fit(
    X_train, 
    y_train_car,
    batch_size=128,
    epochs=10,
    shuffle=True
)

# Evaluate the Network
print("\Traindata : \n")
print(model.evaluate(X_train, y_train_car))

y_test_car = y_test == 1
print("\Testdata : \n")
print(model.evaluate(X_test, y_test_car))
