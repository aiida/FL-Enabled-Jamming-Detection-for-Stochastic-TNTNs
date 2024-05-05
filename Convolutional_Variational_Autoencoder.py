# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:41:02 2024

@author: AP93300
"""


import tensorflow as tf

latent_dim = 10  # Assuming a latent dimension of 10

# Define the encoder network
def encoder_fn(normal_train_CH1):
    input_layer = tf.keras.layers.Input(shape=(normal_train_CH1.shape[1], normal_train_CH1.shape[2], 1))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, activation="relu", padding="same")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

    # Add the latent space
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    x = tf.keras.layers.Dense(8, activation='relu')(x)

    # Define the mean and log variance layers of the latent distribution
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # Define the bottleneck layer of architecture and the sampling layer
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = tf.keras.layers.Lambda(sampling, name='z')([z_mean, z_log_var])

    # Define the encoder model
    encoder = tf.keras.models.Model(input_layer, [z_mean, z_log_var, z], name='encoder')
    return encoder

# Define the decoder network
def decoder_fn():
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(latent_dim, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((5, 9, 4))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(16, (3, 3), activation="relu", strides=2, padding="same")(x)
    x = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Crop the output shape
    decoder_outputs = tf.keras.layers.Cropping2D(cropping=((7, 0), (7, 0)), data_format=None)(x)

    # Define the decoder model
    decoder = tf.keras.models.Model(latent_inputs, decoder_outputs, name='decoder')
    return decoder
