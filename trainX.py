import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
from keras.models import Model, load_model, Input, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPool2D
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Conv2D, MaxPooling2D, Activation, MaxPool1D
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


font0 = {
    'color': 'black',
    'weight': 'bold',
    'size': 10
}

# 먼저 기존의 np.load를 np_load_old에 저장해둠.
np_load_old = np.load

# 기존의 parameter을 바꿔줌
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

np.load.__defaults__ = (None, False, True, 'ASCII')

# load npys
trainData = np.load('/home/bcncompany/PycharmProjects/RCnpys/trainDataX.npy')
trainLabel = np.load('/home/bcncompany/PycharmProjects/RCnpys/trainLabelX.npy')
testData = np.load('/home/bcncompany/PycharmProjects/RCnpys/testDataX.npy')
testLabel = np.load('/home/bcncompany/PycharmProjects/RCnpys/testLabelX.npy')
print(trainData.shape)
print(trainLabel.shape)
print(testData.shape)
print(testLabel.shape)

# gpu setting

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

epochs = 100
batch_size = 5
model_name = 'x5'
model_path = './models/' + model_name + '.h5'
input_shape = (150, 150, 3)

# setting a DataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=30,
    # shear_range=0.2,
    # zoom_range=0.4,
    # horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255
)

train_generator = train_datagen.flow(
    x=trainData, y=trainLabel,
    batch_size=batch_size,
    shuffle=True
)

val_generator = val_datagen.flow(
    x=testData, y=testLabel,
    batch_size=batch_size,
    shuffle=False
)

# model

inputs = Input(shape=(350, 350, 3))

net = Conv2D(8, kernel_size=7, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(16, kernel_size=5, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(256)(net)
net = Activation('relu')(net)
net = Dense(128)(net)
net = Activation('relu')(net)
net = Dense(64)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('linear')(net)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mae')
# model.compile(optimizer='adam', loss=losses.mean_squared_error)
model.summary()

# train

es = EarlyStopping(monitor='val_loss',
                   patience=20,
                   restore_best_weights=True
                   )
mc = ModelCheckpoint(model_path,
                     monitor='val_loss',
                     save_best_only=True, mode='min', verbose=1
                     )
rlrop = ReduceLROnPlateau(monitor='val_loss',
                          patience=3,
                          factor=0.01
                          )

story = model.fit(trainData, trainLabel,
                  batch_size=batch_size,
                  validation_data=(testData, testLabel),
                  shuffle=True,
                  epochs=epochs,
                  callbacks=[es, mc, rlrop]
                  )

# Graphs
# Plot training & validation loss values

plt.plot(story.history['loss'])
plt.plot(story.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.savefig('./graphs/' + model_name + '.png', dpi=300)
plt.show()

