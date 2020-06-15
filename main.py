import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

fashion_dt = keras.datasets.fashion_mnist
(train_im, train_lb), (test_im, test_lb) = fashion_dt.load_data()

# By that list I'll be assigning a concrete name to the numbers (0-9) which are not meaningful to human
labels_list = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# Scaling from 255 pixels. Now only valid pixels be only in range 0-1
train_im = train_im/255.0
test_im = test_im/255.0

# Defining the architecture of the model
model = keras.Sequential([  # Sequential is basically a defining a sequence of layers in order
    keras.layers.Flatten(input_shape=(28, 28)),  # First layer, It flattens multi-dimensional array to only one
    keras.layers.Dense(128, activation='relu'),  # Dense is a fully connected layer. 128 neurons in hidden layer,
    # activation function is rectified linear unit
    keras.layers.Dense(10, activation='softmax')  # 10 nodes in the last layer (labels 0-9)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_im, train_lb, epochs=10)  # epochs determines how many times the model is gonna see the same image from
# that training data provided

'''
# Commented if model is already tested and actual accuracy is known
test_loss, test_acc = model.evaluate(test_im, test_lb, verbose=2)  # We're providing images and labels from the testing
# part to see how the model behaves on these images which it didn't see before

print(test_acc)
'''

prediction = model.predict(test_im)

NEQ_objects = {'color': '#ff2e2e'}
EQ_objects = {'color': '#000000'}


plt.figure(figsize=(10, 10))
for x in range(25):
    plt.subplot(5, 5, x+1)
    plt.imshow(test_im[x], cmap=plt.cm.binary)
    plt.grid(False)
    # Marking on red, if prediction is not the same as the actual value
    if labels_list[test_lb[x]] != labels_list[np.argmax(prediction[x])]:
        plt.title(labels_list[test_lb[x]], fontdict=NEQ_objects)
        plt.xlabel(labels_list[np.argmax(prediction[x])], fontdict=NEQ_objects)
    else:
        plt.title(labels_list[test_lb[x]], fontdict=EQ_objects)
        plt.xlabel(labels_list[np.argmax(prediction[x])], fontdict=EQ_objects)
plt.show()
