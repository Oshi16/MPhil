class Channel(layers.Layer):

    def __init__(self, distance, noise_std=0.1):
        super(Channel, self).__init__()
        self.distance = distance
        self.noise_std = noise_std

    def call(self, signal):
        path_loss = 1 / (self.distance ** 2)
        noise = tf.random.normal(tf.shape(signal), stddev=self.noise_std)
        return signal * path_loss + noise
