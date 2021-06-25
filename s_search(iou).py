import cv2
import numpy as np
from predictor import predict

image = '/home/kaushek/Desktop/SP_PROJECT/images/im2.jpg'

def inter(a,b):
    x11,y11,w1,h1 = a[:]
    x12,y12 = x11+w1, y11+h1
    x21,y21,w2,h2 = b[:]
    x22,y22 = x21+w2,y21+h2
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)
    if x_right < x_left or y_bottom < y_top:
        return 0    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)    
    bb1_area = (h1) * (w1)
    bb2_area = (h2) * (w2)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


cv2.setUseOptimized(True);
cv2.setNumThreads(4);

im = cv2.imread(image)
print(im.dtype)
print(im.shape)
newHeight = 600
newWidth = int(im.shape[1]*600/im.shape[0])
im = cv2.resize(im, (newWidth, newHeight))    

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(im)
ss.switchToSelectiveSearchQuality()
rects = ss.process()

print('Total Number of Region Proposals: {}'.format(len(rects)))

increment = 1
numShowRects = 15

arr = np.zeros((numShowRects,4))
i = 0
for j in range(0,len(rects),1):
    if(i == 0):
        arr[i] = rects[0][:]
        i += 1
    elif(i>=numShowRects):
        break
    else:
        x1,y1,w1,h1 = rects[j][:]
        area_per = ((h1*w1)/(newHeight*newWidth))*100
        if(x1!=0 and y1!=0 and h1!=newHeight and w1!=newWidth):
            flag = 0
            for k in range(0,i,1):
                iou = inter(rects[j][:],arr[k][:])
                if(iou > 0.2):
                    flag = 1
            if (flag == 0 and area_per > 0.5):
                arr[i][0],arr[i][1],arr[i][2],arr[i][3] = rects[j][:]
                i += 1

rec = np.zeros((i,4))
rec = np.int32(rec)
for i in range(0,i,1):
    for j in range(0,4,1):
        rec[i][j] = int(arr[i][j])

k = 0
while (k != 113):    
    imOut = im.copy()
    for i, rect in enumerate(rects):
        if (i < numShowRects):
            x, y, w, h = rect
            ig = im[y:y+h, x:x+w, :]
            ig = cv2.resize(ig, (64, 64))
            img = np.float32(ig)
            acc = predict(img)
            print(i)
            print(acc[0])
            cv2.putText(imOut, str(acc[0][1])+": "+str(acc[0][0])[0:5]+"%", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    cv2.imshow("Output", imOut)
    k = cv2.waitKey(0)
 
    if k == 109:
        numShowRects += increment
    elif k == 108 and numShowRects > increment:
        numShowRects -= increment
    elif k == 113:
        break

cv2.destroyAllWindows()