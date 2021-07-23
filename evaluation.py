import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from PIL import Image
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print(physical_devices)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def func_x(target, ww):
    x = x_model.predict(target)
    crop_x = x[0]
    xx = (ww / 350 * crop_x)
    return xx


def func_y(target, hh):
    y = y_model.predict(target)
    crop_y = y[0]
    yy = (hh / 350 * crop_y)
    return yy


def func_w(target, ww):
    w = w_model.predict(target)
    crop_w = w[0]
    ww = (ww / 350 * crop_w)
    return ww


def func_h(target, hh):
    h = h_model.predict(target)
    crop_h = h[0]
    hh = (hh / 350 * crop_h)
    return hh


def func_x2(target, ww):
    x = x_model2.predict(target)
    crop_x = x[0]
    xx = (ww / 350 * crop_x)
    return xx


def func_y2(target, hh):
    y = y_model2.predict(target)
    crop_y = y[0]
    yy = (hh / 350 * crop_y)
    return yy


def func_w2(target, ww):
    w = w_model2.predict(target)
    crop_w = w[0]
    ww = (ww / 350 * crop_w)
    return ww


def func_h2(target, hh):
    h = h_model2.predict(target)
    crop_h = h[0]
    hh = (hh / 350 * crop_h)
    return hh
# load models
# a_model = load_model('./models/a1.h5')


x_model2 = load_model('/home/bcncompany/PycharmProjects/RCmodels/x5.h5')
y_model2 = load_model('/home/bcncompany/PycharmProjects/optic_ai_server/models/y_model_final.h5')
# y_model2 = load_model('/home/bcncompany/PycharmProjects/RCmodels/y6.h5')
w_model2 = load_model('/home/bcncompany/PycharmProjects/RCmodels/w6.h5')
h_model2 = load_model('/home/bcncompany/PycharmProjects/RCmodels/h6.h5')
x_model = load_model('/home/bcncompany/PycharmProjects/optic_ai_server/models/x_model_final.h5')
y_model = load_model('/home/bcncompany/PycharmProjects/optic_ai_server/models/y_model_final.h5')
w_model = load_model('/home/bcncompany/PycharmProjects/optic_ai_server/models/w_model_final.h5')
h_model = load_model('/home/bcncompany/PycharmProjects/optic_ai_server/models/h_model_final.h5')

# def func_a(target):
#     a = a_model.predict(target)
#     a = a[0]
#     return a



dim2 = (350, 350)  # for predict rotate & crop
# img_path = '/home/bcncompany/PycharmProjects/RCDataset/101/101 (3).JPG'
img_path = '/home/bcncompany/PycharmProjects/neo_classDataset/debug/103-5.jpeg'
img = Image.open(img_path)


# preprocessing (rotate & crop)
original_w, original_h = img.size
print("original_w: " + str(original_w))
print("original_h: " + str(original_h))

predict_target = []
t = np.array(img.resize(dim2))
t = t.astype(np.float32) / 255.
predict_target.append(t)
predict_target = np.array(predict_target, dtype=np.float32)

# predict Rotate & crop
# a = func_a(predict_target)
last_x = func_x(predict_target, original_h)
last_y = func_y(predict_target, original_w)
last_w = func_w(predict_target, original_w)
last_h = func_h(predict_target, original_h)

last_x2 = func_x2(predict_target, original_h)
last_y2 = func_y2(predict_target, original_w)
last_w2 = func_w2(predict_target, original_w)
last_h2 = func_h2(predict_target, original_h)


# convert to numpy(cv2) for rotate & crop
numpy_image = np.array(img)
result = numpy_image[int(last_y): int(last_y) + int(last_h), int(last_x): int(last_x) + int(last_w)]
# result = numpy_image[int(last_y): int(last_y) + int(last_h), int(last_x): int(last_x) + int(last_w+last_w/5)]
result2 = numpy_image[int(last_y2): int(last_y2) + int(last_h2), int(last_x2): int(last_x2) + int(last_w2)]
# show image

f, ax = plt.subplots(1, 3, figsize=(10, 3))
plt.style.use('dark_background')
ax[0].title.set_text('target ')
ax[0].imshow(img)  # first image
ax[0].axis('on')

ax[1].title.set_text('before model')
ax[1].imshow(result)  # mask
ax[1].axis('on')

ax[2].title.set_text('now  model')
ax[2].imshow(result2)  # mask
ax[2].axis('on')
plt.show()

