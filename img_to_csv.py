import numpy as np
import sys
import os
import csv
from PIL import Image

#Useful function
def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


def resize(img):
    size = img.size
    h = size[1]
    w = size[0]
    n_height = 100
    n_width = int((h/n_height)*w)
    re_img = img.resize((n_width,n_height), Image.ANTIALIAS)
    return re_img


myFileList = createFileList('/home/kaushek/Desktop/SP_PROJECT/DATASET/EuroSAT/2750')

for file in myFileList:
    print(file)
    img_file = Image.open(file)
    img = resize(img_file)
    value = np.asarray(img.getdata(), dtype=np.int).reshape(img.size[0],img.size[1],3)
    value = value.flatten()
    print(value)
    with open("river_sample.csv", 'a') as f:
        writer = csv.writer(f)
        writer.writerow(value)