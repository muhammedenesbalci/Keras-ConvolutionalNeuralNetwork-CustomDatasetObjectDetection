# -*- coding: utf-8 -*-
#%% import necessary packages
from module import config
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer  
from sklearn.model_selection import train_test_split
from imutils import paths                         
import matplotlib.pyplot as plt
import numpy as np                                
import pickle
import cv2                                       
import os

# %% Organize CSV file
csvPath = config.IMAGES_ANNOATIONS_CSV_PATH
rows = open(csvPath).read().strip().split("\n")
rows.pop(0)

# %% Loading data from CSV
images = []
targets = []
filenames = []

for row in rows:
    
    try:
        row = row.split(",") 
        label_name, bbox_x, bbox_y ,bbox_width, bbox_height, image_name, image_width, image_height = row
        
        image_width = float(image_width)
        image_height = float(image_height)
        
        startX = float(bbox_x)
        startY = float(bbox_y)
        endX = float(bbox_x) + float(bbox_width)
        endY = float(bbox_y) + float(bbox_height)
        filename = image_name
        
        startX = float(startX) / image_width
        startY = float(startY) / image_height
        
        endX = float(endX) / image_width
        endY = float(endY) / image_height
    
    
        
        img_path = config.IMAGES_PATH + "\\" + filename
        image = load_img(img_path, target_size = (224, 224))
        image = img_to_array(image)
        
        images.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(filename)
        
    except:
        print(f"Memmory Error : {filename}")
    
    
#%%
images = np.array(images, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

#%%
split = train_test_split(images, targets, filenames, test_size=0.10, random_state=42)

(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]


print("[INFO] saving testing filenames...")
f = open(config.TEST_OUTPUT_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

#%%
# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

#%%

opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", metrics = ["accuracy"], optimizer=opt)
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=config.BATCH_SIZE,
	epochs=config.NUM_EPOCHS,
	verbose=1)


#%%
print("[INFO] saving object detector model...")
model.save(config.MODEL_OUTPUT_PATH, save_format="h5")