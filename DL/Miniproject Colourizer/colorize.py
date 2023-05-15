# First Run: pip install -r requirements.txt
# To run code:
# python colorize.py --image | -i relative_image_path
# Example: python colorize.py -i ./images/charlie.jpg

# Import Modules
import numpy as np
import argparse
import cv2
import os
'''
 Training Dataset is computationally expensive and dataset required is quite large.
 Hence we will use Pre-trained Models.
 Model files we used:
 1. colorization_deploy_v2_prototxt.
 2. pts_in_hull.npy.
 3. colorization_release_v2.caffemodel [https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1]
 Place the caffe model in the colorize folder.
'''

# Load Files
DIR = r"./colorize"
PROTOTXT = os.path.join(DIR, r"colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"pts_in_hull.npy")
MODEL = os.path.join(DIR, r"colorization_release_v2.caffemodel")

# Argparser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type = str, required = True, help = "Path to input BW image")
args = vars(ap.parse_args())

# Loading the model
print("Load Model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Loading Centres for AB channel 
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype = "float32")]

# Loading input image
image = cv2.imread(args["image"])
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

# Tune parameters
resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50

# Colorizing
print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis = 2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

# Display Output
cv2.imshow("Original", image)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)
