# -*- coding: utf-8 -*-

#import necessary packages
from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt


print("[INFO] loading object detector...")
model = load_model(config.MODEL_OUTPUT_PATH)
path = "C:\\Users\\enes\\Desktop\\objectDetection\\dataset\\normal\\img\\"
# %%live test
    
import cv2
from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt


# Local cam video capture
cap = cv2.VideoCapture(0)

# Height and width
# First method
width = cap.get(3)
height = cap.get(4)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    succes, frame = cap.read()
    
    if succes == True:
        frame_pred = frame
        
        frame_pred = cv2.resize(frame_pred, (224, 224))
        frame_pred = img_to_array(frame_pred)/ 255.0
        frame_pred = np.expand_dims(frame_pred, axis = 0)
        (boxPreds) = model.predict(frame_pred)
        
        (startX, startY, endX, endY) = boxPreds[0]

        (h,w) = frame.shape[:2]

        startX = int(startX *w)
        startY = int(startY *h)
        endX = int(endX *w)
        endY = int(endY*h)

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame,(startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("video", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Stop capture

cv2.destroyAllWindows()  # Destroy window
    
#%% Images
#import necessary packages
from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os
import matplotlib.pyplot as plt


print("[INFO] loading object detector...")
model = load_model(config.MODEL_OUTPUT_PATH)
pth = "C:\\Users\\enes\\Desktop\\objectDetection\\dataset\\normal\\img\\"

file1 = open(config.TEST_OUTPUT_FILENAMES, 'r')
Lines = file1.readlines()

aa = []
for i in Lines:
    aa.append(i[:-1])

for i in aa:
    image = load_img(pth + i, target_size = (224,224))
    image = img_to_array(image)/ 255.0
    image = np.expand_dims(image, axis = 0)
    
    (boxPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]
    
    
    image = cv2.imread(path + i)
    (h,w) = image.shape[:2]
    
    startX = int(startX *w)
    startY = int(startY *h)
    endX = int(endX *w)
    endY = int(endY*h)
    
    cv2.rectangle(image,(startX, startY), (endX, endY), (0, 255, 0), 15)
    
    plt.figure(), plt.imshow(image, cmap = "gray"), plt.show()
    

#%% Single image
path = "C:\\Users\\enes\\Desktop\\imh.jpg"
image = load_img(path, target_size = (224,224))
image = img_to_array(image)/ 255.0
image = np.expand_dims(image, axis = 0)

(boxPreds) = model.predict(image)
(startX, startY, endX, endY) = boxPreds[0]


image = cv2.imread(path)
(h,w) = image.shape[:2]

startX = int(startX *w)
startY = int(startY *h)
endX = int(endX *w)
endY = int(endY*h)

cv2.rectangle(image,(startX, startY), (endX, endY), (0, 255, 0), 5)

plt.figure(), plt.imshow(image, cmap = "gray"), plt.show()