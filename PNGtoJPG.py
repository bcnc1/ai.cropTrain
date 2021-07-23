# 확장자 일괄 변경
# replace file ext.

import glob
import os.path
import cv2
img_path = '/home/bcncompany/PycharmProjects/RCDataset_inmouth/merge/'
dirs = os.listdir(img_path)
save_path = '/home/bcncompany/PycharmProjects/RCDataset_inmouth/merged/'


def toJpg():
    for item in dirs:
        if os.path.isfile(img_path+item):
            im = cv2.imread(img_path+item)
            f, e = os.path.splitext(item)
            print(save_path+f+'.JPG')
            cv2.imwrite(save_path+f+'.JPG', im)


toJpg()
