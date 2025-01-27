import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

class MessageSplitter(layers.Layer):
    def __init__(self, split_ratio=0.5):
        super(MessageSplitter, self).__init__()
        self.split_ratio = tf.constant(split_ratio, dtype=tf.float32)  # Ratio for splitting into common and private

    def call(self, z1, z2):
        # Ensure inputs are rank 2
        tf.debugging.assert_rank(z1, 2, "z1 must have rank 2 (batch_size, latent_dim)!")
        tf.debugging.assert_rank(z2, 2, "z2 must have rank 2 (batch_size, latent_dim)!")

        # Compute split indices
        latent_dim = tf.shape(z1)[-1]  # Should be the same for z2
        split_idx = tf.cast(self.split_ratio * tf.cast(latent_dim, tf.float32), tf.int32)

        # Validate split index
        tf.debugging.assert_greater(split_idx, 0, "Split index must be positive!")

        # Split z1 and z2
        z1_common, z1_private = z1[:, :split_idx], z1[:, split_idx:]
        z2_common, z2_private = z2[:, :split_idx], z2[:, split_idx:]

        return z1_common, z1_private, z2_common, z2_private

class MessageCombiner(layers.Layer):
    def __init__(self):
        super(MessageCombiner, self).__init__()

    def call(self, z1_common, z2_common):
        z_c = (z1_common + z2_common) #/ 2  # Combine the two common messages
        return z_c

class MessageEncoder(layers.Layer): # Just converting the vectors to signals (z -> s)
    def __init__(self, units):
        super(MessageEncoder, self).__init__()
        self.dense = layers.Dense(units, activation='relu')  # Encoder that transforms z into s

    def call(self, z_common, z_private1, z_private2):
        s_common = self.dense(z_common)
        s_private1 = self.dense(z_private1)
        s_private2 = self.dense(z_private2)

        # Combine signals via superposition
        s_combined = s_common + s_private1 + s_private2

        return tf.reshape(s_combined, (-1, self.dense.units))

class Receiver(layers.Layer):
    def __init__(self, units, user_type):
        super(Receiver, self).__init__()
        self.decoder = Decoder(units) # Decoder to reconstruct the message
        self.user_type = user_type # Store user type as a class variable

    def call(self, s_combined_channel, z_common):
        # Ensure shapes match
        tf.debugging.assert_rank(z_common, 2, "z_common must be rank 2!")
        tf.debugging.assert_rank(s_combined_channel, 2, "s_combined_channel must be rank 2!")

        # Step 1: Decode the common signal
        decoded_common = self.decoder(z_common)

        # Step 2: Remove the common signal from the combined signal (SIC)
        y_sic = tf.reshape(s_combined_channel, tf.shape(decoded_common)) - decoded_common

        # Step 3: Decode the private signal from the residual
        decoded_private = self.decoder(y_sic)

        if self.user_type == 'user1':
            print("Receiver 1: Decoded common and private messages")
        else:
            print("Receiver 2: Decoded common and private messages")

        return decoded_common, decoded_private
