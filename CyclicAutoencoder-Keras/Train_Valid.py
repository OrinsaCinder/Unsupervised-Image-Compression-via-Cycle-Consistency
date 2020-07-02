import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from Models import Encoder, Decoder, BuildAutoencoder
from dataloader import CompressionDataLoader
from tensorflow.keras.callbacks import ModelCheckpoint

Training_Dir = 'Training'
Valid_Dir = 'Validation'

epochs     = 50
batch_size = 8
valid_steps = 5
step_per_epo = 100

TrainDataGen, ValidDataGen = CompressionDataLoader(Training_Dir, Valid_Dir)

Autoencoder = BuildAutoencoder()

#Vae, Encoder, Decoder = build_vae_conv_model()


ModelSaver = ModelCheckpoint('Compression.h5', monitor='TrainingLoss', verbose=1, 
                                                save_best_only=True, 
                                                save_weights_only=False, 
                                                mode='auto')
Autoencoder.fit_generator(TrainDataGen,
                    steps_per_epoch=step_per_epo,
                    epochs= epochs,
                    validation_data=ValidDataGen,
                    validation_steps=valid_steps,
                    callbacks = [ModelSaver])
