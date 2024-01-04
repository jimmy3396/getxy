# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:28:58 2023

@author: 82109
"""

# this code assumes that :
    # we follow option1 when obtaining (p, q)-(x, y) mapping relation
    # we obtain the xy values for (p, q) grid points manually

# for now, i commented all the codes for the left camera to check if 
# the code runs fine with only one camera

import datetime
import numpy as np
import cv2
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.6

coco128 = open('C:\Users\dlgot\desktop\Project_SLM\getxy\coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()
model = YOLO('C:/Users/82109/Desktop/yolov8/yolov8n.pt')

capR = cv2.VideoCapture(0)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#capL = cv2.VideoCapture(2)
#capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

grid_size = 80
rightgridx = open('C:/Users/82109/Desktop/yolov8/rightgridx.txt', 'r')
gridxR = rightgridx.read().split('\n')
rightgridx.close()
rightgridy = open('C:/Users/82109/Desktop/yolov8/rightgridy.txt', 'r')
gridyR = rightgridy.read().split('\n')
rightgridy.close()

gridxR = np.transpose(np.reshape(gridxR, (640//grid_size+1, 480//grid_size+1)).astype(float))
gridyR = np.transpose(np.reshape(gridyR, (640//grid_size+1, 480//grid_size+1)).astype(float))


while True:
    start = datetime.datetime.now()
    
    retR, frameR = capR.read()
#    retL, frameL = capL.read()
    if not retR :
        print('Cam Error')
        break
#    if not retL :
#        print('left cam error')
#        break
    detectionR = model(frameR)[0]
#    detectionL = model(frameL)[0]
    
    for data in detectionR.boxes.data.tolist():
        confidence = float(data[4])
        if confidence < CONFIDENCE_THRESHOLD :
            continue
        xminR, yminR, xmaxR, ymaxR = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        labelR = int(data[5])
        centerxR = (xminR + xmaxR)*0.5
        centeryR = (yminR + ymaxR)*0.5        
        cv2.rectangle(frameR, (xminR, yminR), (xmaxR, ymaxR), (0, 255, 0), 2)
        #cv2.putText(frameR, class_list[labelR]+' '+str(round(confidence, 2))+'%', (xminR, yminR), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
        
        #Bilinear Interpolation
        gridindx = int(centerxR//grid_size)
        gridindy = int(centeryR//grid_size)
        m = centerxR/grid_size - gridindx
        n = centeryR/grid_size - gridindy
        posx = (1-m)*(1-n)*gridxR[gridindy, gridindx]+(1-m)*n*gridxR[gridindy+1, gridindx]+m*(1-n)*gridxR[gridindy, gridindx+1]+m*n*gridxR[gridindy+1, gridindx+1]
        posy = (1-m)*(1-n)*gridyR[gridindy, gridindx]+(1-m)*n*gridyR[gridindy+1, gridindx]+m*(1-n)*gridyR[gridindy, gridindx+1]+m*n*gridyR[gridindy+1, gridindx+1]
        cv2.putText(frameR, 'posx: '+str(posx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frameR, 'posy: '+str(posy), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#    for data in detectionL.boxes.data.tolist():
#        confidence = float(data[4])
#        if confidence < CONFIDENCE_THRESHOLD :
#            continue
#        xminL, yminL, xmaxL, ymaxL = int(data[0]), int(data[1]), int(data[2]), int(data[3])
#        labelL = int(data[5])
#        cv2.rectangle(frameL, (xminL, yminL), (xmaxL, ymaxL), (0, 255, 0), 2)
#        cv2.putText(frameL, class_list[labelL]+' '+str(round(confidence, 2))+'%', (xminL, yminL), cv2.FONT_ITALIC, 1, WHITE, 2)
        
    end = datetime.datetime.now()
    total = (end-start).total_seconds()
    print('Time to process 1 frame: ', total*1000, 'miliseconds')
     
    fps = 'FPS: '+str(1/total)
    cv2.putText(frameR, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#   cv2.putText(frameL, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frameR', frameR)
#    cv2.imshow('frameL', frameL)
    
    if cv2.waitKey(1)==ord('q'):
        break


capR.release()
#capL.release()
cv2.destroyAllWindows()