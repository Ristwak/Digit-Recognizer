import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
""" Loads the MNIST dataset into four variables:
"X_train_full": Training images
"y_train_full": Training labels
"X_test": Test images
"y_test": Test labels  """

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)
# : Splits the training data into training and validation sets. It takes 10% of the training data as the validation set and uses a random seed of 42 for reproducibility

# Normalize pixel values to between 0 and 1
X_train, X_val, X_test = X_train / 255.0, X_val / 255.0, X_test / 255.0
# Normalizes the pixel values of the images to be in the range [0, 1] by dividing them by 255.0. This is a common preprocessing step for neural networks.

#  This is how we can check the value present in X
"""
first_digit_label = y_train[2]
print("Label of the first digit:", first_digit_label) 
image_index = 2
plt.imshow(X_train[image_index], cmap='gray')
plt.title(f"Label: {y_train[image_index]}")
plt.show()
"""

# Create a "graph" detector label
y_train_graph = (y_train == 2)
# Creates a binary label for "graph" detection. It assigns "True" to elements in "y_train" that are equal to 2 and "False" to all other elements
y_val_graph = (y_val == 2)
y_test_graph = (y_test == 2)

# Build a CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Binary classification for "graph" or "not graph"
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Compiles the model, specifying the optimizer, loss function, and evaluation metric. It uses the Adam optimizer and binary cross-entropy loss for binary classification.

# Train the model
history = model.fit(X_train.reshape(-1, 28, 28, 1), y_train_graph, epochs=5, validation_data=(X_val.reshape(-1, 28, 28, 1), y_val_graph))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test_graph)
# Evaluates the trained model on the test data and calculates the test loss and accuracy.
print("\nTest accuracy:", test_acc)

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()