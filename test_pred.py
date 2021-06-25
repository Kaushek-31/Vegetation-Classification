import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers. normalization import BatchNormalization
import numpy as np
from pandas import read_csv
import cv2
import pickle
import joblib

testset = '/home/kaushek/Desktop/SP_PROJECT/TEST_CODES/test_set.csv'
test_csv = read_csv(testset)


def create_tag_mapping(train_csv):
	labels = set()
	for i in range(len(train_csv)):
		tags = train_csv['y'][i].split(' ')
		labels.update(tags)
	labels = list(labels)
	labels.sort()
	labels_map = {labels[i]:i for i in range(len(labels))}
	inv_labels_map = {i:labels[i] for i in range(len(labels))}
	return labels_map, inv_labels_map

labels, ind_labels = create_tag_mapping(test_csv)

x_test = []
y_test = np.zeros((len(test_csv),10))

for i in range(len(test_csv)):
    img = cv2.imread(test_csv['x'][i])
    print("uploaded:" + str(i+1))
    x_test.append(np.array(img, dtype = float))
    txt = str(test_csv['y'][i].split(' '))
    txt = txt[2:len(txt)-2]
    print(txt)
    k = labels[txt]
    print(k)
    y_test[i][k] = 1

size = 5000
pred_x = np.float64(x_test[0:size])

acc_array = np.zeros((size,2))
acc_array = np.float32(acc_array)
model = joblib.load('/home/kaushek/satellite_model.pkl')
preds = model.predict(pred_x, verbose=1)

for i in range(0,len(preds),1):
    max = 0
    for j in range(0,10,1):
        if(preds[i][j]>max):
            max = preds[i][j]
            
    for j in range(0,10,1):
        if(preds[i][j] == max):
            preds[i][j] = 1
            acc_array[i][0] = j
            acc_array[i][1] = max*100
        else:
            preds[i][j] = 0

count = 0
pred_y = y_test[0:size]

for i in range(0,len(preds),1):
    flag = 0
    for j in range(0,10,1):
        if preds[i][j] != pred_y[i][j]:
            flag = 1
            break
        else:
            flag = 0
    if flag == 0:
        count += 1

acc = (count/len(preds))*100
print("THE ACCURACY ON TESTING "+str(size)+" SAMPLES IS: "+str(acc))
print("\n.............PREDICTION RESULTS..............")
print(acc_array)