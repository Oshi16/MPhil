import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

import VAEWithRSMA

def train_vae_with_rsma(X_train, vae_with_rsma, learning_rate=1e-3, num_epochs=2, batch_size=32):
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Iterate through the training data in batches
        for i in range(0, len(X_train), batch_size):
            batch = X_train[i:i+batch_size]

            # Here, each batch is split into two sets: one for input_1 and one for input_2
            for j in range(0, batch.shape[0], 2):
                if j+1 >= batch.shape[0]:
                    continue

                input_1 = batch[j]    # First image of the pair
                input_2 = batch[j+1]  # Second image of the pair

                print(f"Training inputs: input_1 shape: {input_1.shape}, input_2 shape: {input_2.shape}")

                input_1 = tf.expand_dims(input_1, axis=0)  # Expand dims to match batch input shape
                input_2 = tf.expand_dims(input_2, axis=0)

                with tf.GradientTape() as tape:
                    # Pass the two images through the system
                    decoded_common_1, decoded_private_1, decoded_common_2, decoded_private_2 = vae_with_rsma((input_1, input_2))

                    # define reconstructed signals
                    reconstructed_1 = decoded_common_1 + decoded_private_1  # Reconstructed signal from user 1
                    reconstructed_2 = decoded_common_2 + decoded_private_2  # Reconstructed signal from user 2

                    # mu and log_var from the encoder layers if needed.
                    mu1, log_var1 = vae_with_rsma.encoder(input_1)
                    mu2, log_var2 = vae_with_rsma.encoder(input_2)

                    # Compute the VAE loss
                    loss_value = vae_loss(input_1, reconstructed_1, mu1, log_var1, input_2, reconstructed_2, mu2, log_var2)

                # Update the model parameters
                grads = tape.gradient(loss_value, vae_with_rsma.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae_with_rsma.trainable_weights))

                total_loss += loss_value
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
