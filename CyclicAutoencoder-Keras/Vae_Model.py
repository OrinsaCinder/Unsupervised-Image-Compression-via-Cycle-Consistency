from __future__ import print_function

import numpy as np

from keras.layers import Layer, Input, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose, Subtract
from keras.models import Model
from keras import backend as K
from keras.losses import mse, binary_crossentropy


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var, scale_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon * scale_var


def build_vae_encoder_model(filters      = 32, 
                            latent_dims  = 2, 
                            z_activation = 'relu'):
    #input_shape = (28, 28, 1)
    input_shape = (150, 150, 1)
    input_img = Input(shape=input_shape) # the input image
    input_scale_var = Input(shape=(1,))        # used to scale z_log_var
    x         = Conv2D(filters, (3,3), activation='relu', strides=(2,2), padding='same', name='conv0_encode')(input_img)
    x         = Conv2D(filters*2, (3,3), activation='relu', strides=(2,2), padding='same', name='conv1_encode')(x)
    x         = Flatten(name='flatten0_encode')(x)
    x         = Dense(32, activation='relu', name='dense0_encode')(x)
    z_mean    = Dense(latent_dims, name='z_mean_encode')(x)
    z_log_var = Dense(latent_dims, name='z_log_var_encode')(x)
    z         = Lambda(sampling, output_shape=(latent_dims,), name='z_encode')([z_mean, z_log_var, input_scale_var])
    encoder   = Model([input_img, input_scale_var], [z, z_mean, z_log_var], name='encoder')
    encoder.summary()
    return encoder


def build_vae_decoder_model(filters     = 32, 
                            latent_dims = 2, 
                            output_cnls = 1):
    z_input = Input(shape=(latent_dims,))
    x = Dense(7*7*filters*2, activation='relu')(z_input)
    x = Reshape((7, 7, filters*2))(x)
    x = Conv2DTranspose(filters*2, (3,3), activation='relu', strides=(2,2), padding='same', name='conv0_decode')(x)
    x = Conv2DTranspose(filters, (3,3), activation='relu', strides=(2,2), padding='same', name='conv1_decode')(x)
    decoded = Conv2DTranspose(1, (3,3), activation='sigmoid', padding='same', name='output')(x)
    decoder = Model(z_input, decoded, name='decoder')
    decoder.summary()
    return decoder


def build_vae_conv_model():
    """ build a convolutional variational autoencoder. """
    inputs_img   = Input(shape=(150, 150, 1))
    inputs_scale_var = Input(shape=(1,))
    encoder  = build_vae_encoder_model()
    decoder  = build_vae_decoder_model()
    z, z_mean, z_log_var  = encoder([inputs_img, inputs_scale_var])
    outputs  = decoder(z)
    #print()
    vae = Model([inputs_img, inputs_scale_var], [outputs], name='vae')
    # define custom loss
    reconstruction_loss = mse(K.flatten(inputs_img), K.flatten(outputs))
    reconstruction_loss *= (28*28)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    return vae, encoder, decoder


