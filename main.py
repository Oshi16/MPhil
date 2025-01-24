import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values (from 0-255 to 0-1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images from (28, 28) to (784,)
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Check the shapes of the data
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

vae_with_rsma = VAEWithRSMA(original_dim=784, latent_dim=32, distance=10, noise_std=0.5, common_ratio=0.5)
train_vae_with_rsma(X_train, vae_with_rsma)
