import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualize satellite images
from skimage.io import imshow # visualize satellite images

x_train_set_fpath = '/home/kaushek/Desktop/SP_PROJECT/test_sample.csv'

print ('Loading Training Data')
X_train = pd.read_csv(x_train_set_fpath)
print ('Loaded 10 x 10 x 4 images')
print(X_train)
X_train = X_train.as_matrix()

print ('We have',X_train.shape[0],'examples and each example is a list of',X_train.shape[1],'numbers.')

print (X_train.shape)

