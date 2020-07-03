import csv
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def read_csv():
    ### This function reads image paths from csv file ###

    samples = []
    with open('./data/my_driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def split_samples(samples):
    ### This function splits the samples into training and validation samples ###

    # Splitting samples - 80% as training data and 20% as validation data
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def read_batch_samples(batch_samples):
    ### This function reads the center images and the steering angle batch by batch and flip them ###

    center_images = []
    angles = []

    for batch_sample in batch_samples:
        source_path = batch_sample[0]
        file_name = source_path.split('/')[-1]
        current_path='./data/IMG/' + file_name
        center_image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
        angle = float(batch_sample[3])

        # Flip images and steering angles
        image_flipped = np.fliplr(center_image)
        angle_flipped = -angle

        center_images.append(center_image)
        angles.append(angle)

    return center_images, angles

def generator(samples, batch_size=32):
    ### This function generate training or validation data with helf of python generator ###

    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            center_images, angles = read_batch_samples(batch_samples)

            X_train = np.array(center_images)
            y_train = np.array(angles)

            yield shuffle(X_train, y_train)

def cnn():
    ### The function preprocess the data and creates the NVIDIA cnn model ###

    # Preprocess data
    model = Sequential()
    # Normalize the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop the data
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    # NVIDIA CNN model
    model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
    model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
    model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(Conv2D(64,(3,3),activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    return model

batch_size = 32

samples = read_csv()

X_train, X_valid = split_samples(samples)

train_generator = generator(X_train, batch_size=batch_size)

validation_generator = generator(X_valid, batch_size=batch_size)

model = cnn()

model.compile(optimizer='adam', loss='mse')

model.summary()

history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(X_train)/batch_size),    validation_data=validation_generator, validation_steps=math.ceil(len(X_valid)/batch_size), epochs=5,    verbose=1)

model.save('model.h5')

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
