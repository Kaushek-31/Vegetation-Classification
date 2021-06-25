import numpy as np
import sys
import os
import csv
from PIL import Image

#Useful function
def createFileList(myDir):
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

myFileList = createFileList('/home/kaushek/Desktop/SP_PROJECT/DATASET/EuroSAT/test/SeaLake')


with open("test_set.csv", 'a') as f:
    writer = csv.writer(f)
    for i in range(0,len(myFileList),1):
        writer.writerow([myFileList[i],'Sea/Lake'])