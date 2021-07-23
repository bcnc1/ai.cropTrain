import glob
from PIL import Image

ImageList = glob.glob("/home/bcncompany/PycharmProjects/RCDataset_inmouth/merge150/*.JPG")

for img_path in ImageList:
    print(img_path)
    img = Image.open(img_path)
    img.resize((150, 150)).save(img_path)
