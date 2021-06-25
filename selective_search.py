import cv2
import numpy as np

cv2.setUseOptimized(True);
cv2.setNumThreads(4);

im = cv2.imread('/home/kaushek/Desktop/SP_PROJECT/images/im2.jpg')
newHeight = 600
newWidth = int(im.shape[1]*600/im.shape[0])
im = cv2.resize(im, (newWidth, newHeight))    

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

ss.setBaseImage(im)
c = input()

if (c == 'f'):
    ss.switchToSelectiveSearchFast()

elif (c == 'q'):
    ss.switchToSelectiveSearchQuality()
elif (c == 's'):
    ss.switchToSingleStrategy()
    
rects = ss.process()
print('Total Number of Region Proposals: {}'.format(len(rects)))

numShowRects = 400
increment = 1

arr = np.zeros((numShowRects,4))
i = 0
barr = 50
for j in range(0,len(rects),1):
    if(i == 0):
        arr[i] = rects[0][:]
        i += 1
    elif(i>=numShowRects):
        break
    else:
        x1,y1,w1,h1 = rects[j][:]
        if(x1!=0 and y1!=0 and h1!=newHeight and w1!=newWidth):
            for k in range(0,i,1):
                flag1,flag2 = 0,0
                x2,y2,w2,h2 = arr[k][:]
                xb,yb,hb,wb = x1-x2,y1-y2,h2-h1,w2-w1
                if(xb<-150 and yb<-150):
                    if(wb>xb and hb>yb):
                        flag2 = 1
                if(xb<150 and yb<150):
                    if(wb<175 and hb<175):
                        flag1 = 1
                if(flag1 == 0 and flag2==0):
                    arr[i][0],arr[i][1],arr[i][2],arr[i][3] = x1,y1,w1,h1
                    i += 1
                    break
                if(flag2 == 1):
                    arr[k][0],arr[k][1],arr[k][2],arr[k][3] = x1,y1,w1,h1
                    break

rec = np.zeros((i,4))
rec = np.int32(rec)
for i in range(0,i,1):
    for j in range(0,4,1):
        rec[i][j] = int(arr[i][j])

while True:    
    imOut = im.copy()
    for i, rect in enumerate(rec):
        if (i < numShowRects):
            x, y, w, h = rect
            cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    cv2.imshow("Output", imOut)
 
    k = cv2.waitKey(10) & 0xFF
 
    if k == 109:
        numShowRects += increment
    elif k == 108 and numShowRects > increment:
        numShowRects -= increment
    elif k == 113:
        break

cv2.destroyAllWindows()