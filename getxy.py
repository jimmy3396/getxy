# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 2023

@author: jimmy lee
"""

# This code is for obtaining (x, y) values of detected objects using YOLOv5(optional : SAM(Spatial Attention Module))
# this code assumes that :
    # we follow option1 when obtaining (p, q)-(x, y) mapping relation
    # we obtain the xy values for (p, q) grid points manually
    # (p, q) grid points are evenly spaced
    # (x, y) grid points will be stored in a text file in the following format: "object_positions.txt"
    # image file "rightgrid.png" will inform SLM what to make pixel image
# for now, I commented all the codes for the left camera to check if 
# the code runs fine with only one camera

import datetime
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO # optional : SAM(Spatial Attention Module)
from scipy.stats import multivariate_normal
from diffractio import um, mm, degrees, np, plt
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from slmsuite.holography.algorithms import Hologram

# Path to the frame_image folder
frame_image_folder = 'C:/study/Project_SLM/getxy/frame_image/' # empty the folder before running the code

# Remove all files in the folder
file_list = os.listdir(frame_image_folder)
for file_name in file_list:
    file_path = os.path.join(frame_image_folder, file_name)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# Set the parameters for the SLM phase mask
z = 250*mm
wavelength = 0.5 * um
xin = np.linspace(-1*mm, 1*mm, 640)
yin = np.linspace(-1*mm, 1*mm, 480)

xout = np.linspace(-10*mm, 10*mm, 1024)
yout = np.linspace(-10*mm, 10*mm, 1024)


# Set Yolov8 position detection constant        
CONFIDENCE_THRESHOLD = 0.6

coco128 = open('C:/study/Project_SLM/getxy/coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()
model = YOLO('C:/study/Project_SLM/getxy/yolov8n.pt')

capR = cv2.VideoCapture(0)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

grid_size = 80
rightgridx = open('C:/study/Project_SLM/getxy/rightgridx.txt', 'r')
gridxR = rightgridx.read().split('\n')
rightgridx.close()
rightgridy = open('C:/study/Project_SLM/getxy/rightgridy.txt', 'r')
gridyR = rightgridy.read().split('\n')
rightgridy.close()

gridxR = np.transpose(np.reshape(gridxR, (640//grid_size+1, 480//grid_size+1)).astype(float))
gridyR = np.transpose(np.reshape(gridyR, (640//grid_size+1, 480//grid_size+1)).astype(float))

with open('C:/study/Project_SLM/getxy/object_positions.txt', 'w') as output_file:
    pass  # This line truncates the file, making it empty

frame_counter = 0

while True:
    start = datetime.datetime.now()
    
    retR, frameR = capR.read()
    if not retR :
        print('Cam Error')
        break

    detectionR = model(frameR)[0]
    output_file_path = 'C:/study/Project_SLM/getxy/object_positions.txt'
    
    # Create an empty image to draw circles
    img_shape = (480, 640) # (height, width, channel) should be (1200, 1920) for 4K
    img = np.zeros(img_shape, dtype=np.uint8)

    
    with open(output_file_path, 'a') as output_file:
        output_file.write("\n--------------------------------------------\n")
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
            # Write posx and posy data to the file
            output_file.write(f'{labelR}: posx {posx}, posy {posy}, ')
            
            # Draw a circle with Gaussian distribution
            #sigma = 5  # Standard deviation for Gaussian distribution
            #x, y = np.meshgrid(np.arange(0, img_shape[1]), np.arange(0, img_shape[0]))
            #pos = np.empty(x.shape + (2,))
            #pos[:, :, 0] = x
            #pos[:, :, 1] = y
            #rv = multivariate_normal([centerxR, centeryR], cov=[[sigma, 0], [0, sigma]])
            
            # Draw a circle on the generated image
            cv2.circle(img, (int(centerxR), int(centeryR)), radius=15, color=(255, 255, 255), thickness=-1)
            
            # Intensity inside the circle is a Gaussian distribution
            #img += (rv.pdf(pos) * 255).astype(np.uint8)

            cv2.putText(frameR, 'posx: '+str(posx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frameR, 'posy: '+str(posy), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Save the generated image to a file with the frame number in the filename
    img_file_path = f'C:/study/Project_SLM/getxy/frame_image/generated_image_frame_{frame_counter}.png'
    cv2.imwrite(img_file_path, img)

    frame_counter += 1  # Increment frame counter for the next frame
    
    end = datetime.datetime.now()
    total = (end-start).total_seconds()
    print('Time to process 1 frame: ', total*1000, 'miliseconds')
     
    fps = 'FPS: '+str(1/total)
    cv2.putText(frameR, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('frameR', frameR)
    
    if cv2.waitKey(1)==ord('q'):
        break


capR.release()
cv2.destroyAllWindows()