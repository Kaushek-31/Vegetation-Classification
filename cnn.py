import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from pandas import read_csv
import cv2
from random import randint

testset = '/home/kaushek/Desktop/SP_PROJECT/TEST_CODES/test_set.csv'
trainset = '/home/kaushek/Desktop/SP_PROJECT/TEST_CODES/train_set.csv'

train_csv = read_csv(trainset)
test_csv = read_csv(testset)
img_size = 64
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

labels, ind_labels = create_tag_mapping(train_csv)
print(labels)
print(ind_labels)

x_train = []
y_train = np.zeros((len(train_csv),10))

for i in range(len(train_csv)):
    img = cv2.imread(train_csv['x'][i])
    print("uploaded:" + str(i+1))
    x_train.append(np.array(img, dtype = float))
    txt = str(train_csv['y'][i].split(' '))
    txt = txt[2:len(txt)-2]
    k = labels[txt]
    y_train[i][k] = 1

x_train = np.float32(x_train)
print("done conversion")

x_train = x_train/255
print("done normalization")

model = Sequential()
model.add(Conv2D(64, kernel_size = (3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.01)

x_test = []
y_test = np.zeros((len(test_csv), 10))
    
for i in range(len(test_csv)):
    img = cv2.imread(test_csv['x'][i])
    print("uploaded:" + str(i+1))
    x_test.append(np.array(img, dtype = float))
    txt = str(test_csv['y'][i].split(' '))
    txt = txt[2:len(txt)-2]
    k = labels[txt]
    y_test[i][k] = 1

test_x = []
test_y = []
pred_length = 10

for i in range(0,pred_length,1):
    r = randint(0,len(test_csv))
    test_x.append(np.float32(x_test[r]))
    test_y.append(y_test[r])

preds = []
for i in range(pred_length):
    test_x[i] = test_x[i]/255
    preds.append(model.predict(test_x[i], verbose=1))


print(preds)  
print(test_y)