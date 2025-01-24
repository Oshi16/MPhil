# VAEWithRSMA model
class VAEWithRSMA(tf.keras.Model):
    def __init__(self, original_dim, latent_dim, distance, noise_std, common_ratio):
        super(VAEWithRSMA, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.sampling = Sampling()
        self.splitter = MessageSplitter(split_ratio=common_ratio)
        self.combiner = MessageCombiner()
        self.message_encoder = MessageEncoder(units=latent_dim)
        self.channel = Channel(distance)
        self.receiver1 = Receiver(latent_dim, user_type='user1')
        self.receiver2 = Receiver(latent_dim, user_type='user2')

        self.decoder = Decoder(original_dim)

    def call(self, inputs):
        input_1, input_2 = inputs
        '''# Expand dims to match batch input shape (required for VAE)
        input_1 = tf.expand_dims(input_1, axis=0)
        input_2 = tf.expand_dims(input_2, axis=0)'''

        # Encode inputs into latent space
        mu1, log_var1 = self.encoder(input_1)
        mu2, log_var2 = self.encoder(input_2)

        # Sample the latent vectors z1 and z2
        z1 = self.sampling((mu1, log_var1))
        z2 = self.sampling((mu2, log_var2))

        # Reshape z1 and z2 to ensure they are rank 2 (batch_size, latent_dim)
        z1 = tf.reshape(z1, (-1, tf.shape(z1)[-1]))  # Shape: (batch_size, latent_dim)
        z2 = tf.reshape(z2, (-1, tf.shape(z2)[-1]))

        # Split the messages into common and private parts
        z1_common, z1_private, z2_common, z2_private = self.splitter(z1, z2)

        # Combine the common parts into a single common message
        z_common = self.combiner(z1_common, z2_common)

        # Encode the common and private messages into signals
        s_combined = self.message_encoder(z_common, z1_private, z2_private)

        # Transmit the signals through the free-space channel
        s_combined_channel = self.channel(s_combined)

        # Simulate two receivers
        decoded_common_1, decoded_private_1 = self.receiver1(s_combined_channel, z_common)
        decoded_common_2, decoded_private_2 = self.receiver2(s_combined_channel, z_common)

        print(f"Shapes: input_1: {input_1.shape}, input_2: {input_2.shape}, z1: {z1.shape}, z2: {z2.shape}")
        print(f"z_common: {z_common.shape}, s_combined: {s_combined.shape}, s_combined_channel: {s_combined_channel.shape}")

        return decoded_common_1, decoded_private_1, decoded_common_2, decoded_private_2

# Loss function for the VAE
def vae_loss(inputs1, outputs1, mu1, log_var1, inputs2, outputs2, mu2, log_var2):
    # Reshape outputs to match inputs
    outputs1 = tf.reshape(outputs1, tf.shape(inputs1))
    outputs2 = tf.reshape(outputs2, tf.shape(inputs2))

    reconstruction_loss_1 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs1, outputs1))
    reconstruction_loss_1 *= inputs1.shape[1]

    reconstruction_loss_2 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs2, outputs2))
    reconstruction_loss_2 *= inputs2.shape[1]

    kl_loss_1 = -0.5 * tf.reduce_mean(1 + log_var1 - tf.square(mu1) - tf.exp(log_var1))
    kl_loss_2 = -0.5 * tf.reduce_mean(1 + log_var2 - tf.square(mu2) - tf.exp(log_var2))

    return reconstruction_loss_1 + kl_loss_1 + reconstruction_loss_2 + kl_loss_2
