import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, ReLU, LeakyReLU, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MSE
from tensorflow.keras import backend as K
from tensorflow import reshape

def Encoder():
    input_img = Input(shape = (150, 150, 1))
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (1, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(2, (1, 1), activation='relu', padding='same')(x)
    Coding = Conv2D(1, (1, 1), activation='relu', padding='same')(x)
    encoder = Model(input_img, Coding, name="encoder_model")
    encoder.summary()
    print(Coding.shape)
    return encoder

def Decoder():
    Coding = Input(shape=(7,7,1))
    x = Conv2D(2, (1, 1), activation='relu', padding='same')(Coding)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    Decoded = Conv2D(1, (3, 3), activation='tanh', padding='same')(x)
    decoder = Model(Coding, Decoded, name='decoder_model')
    decoder.summary()
    return decoder

# Lcycle(I, I') = Lmse(Ef (I), Ef (I'))
'''
def CyclicLoss(input_img):
    InitCoding = Encoder(input_img)
    ReconCoding = Encoder(Decoder(Encoder(input_img)))
    CyclicLoss = MSE(InitCoding, ReconCoding)
    return CyclicLoss
'''

def CompressingLoss(coding, alpha = 0.5):
    CodingL1 = K.abs(coding)
    #CodingL2 = K.sum(K.square(coding))
    #CompressLoss = CodingL1/CodingL2 + alpha*CodingL2/CodingL1
    return CodingL1


def BuildAutoencoder(lambda1 = 0.1, lambda2 = 0.05):
    encoder = Encoder()
    decoder = Decoder()
    input_img = Input(shape=(150, 150, 1))
    #print(type(input_img))

    Coding_Input = encoder(input_img)
    print(Coding_Input.shape)
    output_img = decoder(Coding_Input)
    Coding_Output = encoder(decoder(Coding_Input))
    print(Coding_Output.shape)

    CyclicLoss = MSE(Coding_Input,Coding_Output)
    print(CyclicLoss)

    CompressLoss_0 = CompressingLoss(Coding_Input)
    CompressLoss = reshape(CompressLoss_0,[CompressLoss_0.shape[1],CompressLoss_0.shape[2]])
    print(CompressLoss)

    TrainingLoss = lambda1*CompressLoss + lambda2*CyclicLoss

    Autoencoder = Model(input_img, output_img, name = 'AutoEncoder')
    Autoencoder.add_loss(TrainingLoss)
    Autoencoder.compile(optimizer='adam')
    Autoencoder.summary()
    return Autoencoder