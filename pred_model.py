import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers. normalization import BatchNormalization
import numpy as np
from pandas import read_csv
import cv2
import pickle

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
x_test = []
y_train = np.zeros((len(train_csv),10))
y_test = np.zeros((len(test_csv),10))

for i in range(len(train_csv)):
    img = cv2.imread(train_csv['x'][i])
    print("uploaded:" + str(i+1))
    x_train.append(np.array(img, dtype = float))
    txt = str(train_csv['y'][i].split(' '))
    txt = txt[2:len(txt)-2]
    k = labels[txt]
    y_train[i][k] = 1

for i in range(len(test_csv)):
    img = cv2.imread(test_csv['x'][i])
    print("uploaded:" + str(i+1))
    x_test.append(np.array(img, dtype = float))
    txt = str(test_csv['y'][i].split(' '))
    txt = txt[2:len(txt)-2]
    k = labels[txt]
    y_test[i][k] = 1


X_train = np.asarray(x_train)
X_test = np.asanyarray(x_test)
X_train /= 255
X_test /= 255


batch_size = 50
num_classes = 10
epochs = 75
num_predictions = 20
img_size = 64

model = Sequential()
model.add(Conv2D(64, kernel_size = (4, 4), activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size = (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size = (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size= (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_split=0.01)
print("................TRAINING DONE....................")
size = 5000
pred_x = np.float64(x_test[0:size])/255
preds = model.predict(pred_x, verbose = 1)
acc_array = np.zeros((size,2))
acc_array = np.float32(acc_array)

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

filename = '/home/kaushek/satellite_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
print(".............Saving model Done................")