import joblib
import pickle
import numpy as np
from pandas import read_csv

def predict(x_test):
    
    x_test = x_test.reshape(-1,x_test.shape[0],x_test.shape[1],x_test.shape[2])
    filename = '/home/kaushek/satellite_model.pkl'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    test_data = read_csv('/home/kaushek/Desktop/SP_PROJECT/TEST_CODES/test_set.csv')
    labels = set()
	
    for i in range(len(test_data)):
        tags = test_data['y'][i].split(' ')
        labels.update(tags)
	
    labels = list(labels)
    labels.sort()
    
    inv_labels_map = {i:labels[i] for i in range(len(labels))}
    
    pred_x = np.float64(x_test)/255
    preds = model.predict(pred_x, verbose = 1)
    accr = []
    
    for i in range(0,len(preds),1):
        max = 0
        acc_array = []
        for j in range(0,10,1):
            if(preds[i][j]>max):
                max = preds[i][j]
                
        for j in range(0,10,1):
            if(preds[i][j] == max):
                preds[i][j] = 1
                acc_array.append(max*100)
                acc_array.append(inv_labels_map[j])
                accr.append(acc_array)
            else:
                preds[i][j] = 0

    return(accr)
