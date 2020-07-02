import os
from skimage import io
from keras.preprocessing.image import ImageDataGenerator

def CompressionDataLoader(train_dir, valid_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),batch_size=8, class_mode = 'binary')
    valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(150, 150),batch_size=8, class_mode = 'binary')
    #for train_batch in train_generator:
        #print('training_img:', train_batch[0].shape)
    return train_generator, valid_generator