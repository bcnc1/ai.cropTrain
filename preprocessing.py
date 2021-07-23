import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from skimage.transform import resize
import sys, glob
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Add, LeakyReLU
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras import losses
import keras

from tensorflow.keras import datasets, layers, models
import datetime

import matplotlib.image as img
import matplotlib.pyplot as pp
from PIL import Image

labels = pd.read_excel('./Labeling_inmouth.xlsx', sheet_name='800', engine='openpyxl')
labels_np = labels.values
print(labels.describe())
print('\n\n')

imgs_npy = 'imgs'

trainDataA_npy = 'trainDataA'
testDataA_npy = 'testDataA'
trainDataX_npy = 'trainDataX'
testDataX_npy = 'testDataX'
trainDataY_npy = 'trainDataY'
testDataY_npy = 'testDataY'
trainDataW_npy = 'trainDataW'
testDataW_npy = 'testDataW'
trainDataH_npy = 'trainDataH'
testDataH_npy = 'testDataH'


trainLabelA_npy = 'trainLabelA'
testLabelA_npy = 'testLabelA'
trainLabelX_npy = 'trainLabelX'
testLabelX_npy = 'testLabelX'
trainLabelY_npy = 'trainLabelY'
testLabelY_npy = 'testLabelY'
trainLabelW_npy = 'trainLabelW'
testLabelW_npy = 'testLabelW'
trainLabelH_npy = 'trainLabelH'
testLabelH_npy = 'testLabelH'




labels = pd.read_csv('./Labeling_inmouth.txt', sep="\t", header=None)
labels_np = labels.values
print(labels.describe())

img_path = '/home/bcncompany/PycharmProjects/RCDataset_inmouth/merged/'
imgs = np.empty((len(labels_np[:, 0]), 350, 350, 3), dtype=object)

for i, img_filename in enumerate(labels_np[:, 0]):
    img = os.path.join(img_path, img_filename)
    print(img_filename)
    res = plt.imread(img + ".JPG")
    # res = resize(img, (350, 350))
    imgs[i] = res

# X-Position

trainData, testData, trainLabel, testLabel = train_test_split(imgs, labels_np[:, 2], test_size=0.1)
print("X-Position")
print(trainData.shape, trainLabel.shape)
print(testData.shape, testLabel.shape)
trainData = np.asarray(trainData).astype('float32')
testData = np.asarray(testData).astype('float32')
trainLabel = np.asarray(trainLabel).astype('float32')
testLabel = np.asarray(testLabel).astype('float32')
np.save('./npys/' + trainDataX_npy, trainData)
np.save('./npys/' + testDataX_npy, testData)
np.save('./npys/' + trainLabelX_npy, trainLabel)
np.save('./npys/' + testLabelX_npy, testLabel)

# Y-Position

trainData, testData, trainLabel, testLabel = train_test_split(imgs, labels_np[:, 3], test_size=0.1)
print("Y-Position")
print(trainData.shape, trainLabel.shape)
print(testData.shape, testLabel.shape)
trainData = np.asarray(trainData).astype('float32')
testData = np.asarray(testData).astype('float32')
trainLabel = np.asarray(trainLabel).astype('float32')
testLabel = np.asarray(testLabel).astype('float32')
np.save('./npys/' + trainDataY_npy, trainData)
np.save('./npys/' + testDataY_npy, testData)
np.save('./npys/' + trainLabelY_npy, trainLabel)
np.save('./npys/' + testLabelY_npy, testLabel)

# Weight

trainData, testData, trainLabel, testLabel = train_test_split(imgs, labels_np[:, 4], test_size=0.1)
print("weight")
print(trainData.shape, trainLabel.shape)
print(testData.shape, testLabel.shape)
trainData = np.asarray(trainData).astype('float32')
testData = np.asarray(testData).astype('float32')
trainLabel = np.asarray(trainLabel).astype('float32')
testLabel = np.asarray(testLabel).astype('float32')
np.save('./npys/' + trainDataW_npy, trainData)
np.save('./npys/' + testDataW_npy, testData)
np.save('./npys/' + trainLabelW_npy, trainLabel)
np.save('./npys/' + testLabelW_npy, testLabel)

# Height

trainData, testData, trainLabel, testLabel = train_test_split(imgs, labels_np[:, 5], test_size=0.1)
print("height")
print(trainData.shape, trainLabel.shape)
print(testData.shape, testLabel.shape)
trainData = np.asarray(trainData).astype('float32')
testData = np.asarray(testData).astype('float32')
trainLabel = np.asarray(trainLabel).astype('float32')
testLabel = np.asarray(testLabel).astype('float32')
np.save('./npys/' + trainDataH_npy, trainData)
np.save('./npys/' + testDataH_npy, testData)
np.save('./npys/' + trainLabelH_npy, trainLabel)
np.save('./npys/' + testLabelH_npy, testLabel)

# Angle

trainData, testData, trainLabel, testLabel = train_test_split(imgs, labels_np[:, 1], test_size=0.1)
print("Angle")
print(trainData.shape, trainLabel.shape)
print(testData.shape, testLabel.shape)
trainData = np.asarray(trainData).astype('float32')
testData = np.asarray(testData).astype('float32')
trainLabel = np.asarray(trainLabel).astype('float32')
testLabel = np.asarray(testLabel).astype('float32')
np.save('./npys/' + trainDataA_npy, trainData)
np.save('./npys/' + testDataA_npy, testData)
np.save('./npys/' + trainLabelA_npy, trainLabel)
np.save('./npys/' + testLabelA_npy, testLabel)
