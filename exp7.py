#IPMV all codes (you can delete these green comments dont worry:) )
# copy this segment in the begenning of all the experiments (mandatory do not fuck up!)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn import datasets, svm, metrics, cluster, naive_bayes
from sklearn.model_selection import train_test_split
from skimage import feature, measure, morphology, segmentation, exposure, filters, img_as_float
import urllib.request
import os
print("All required libraries imported successfully!")
 # download a picture from google place it in google colab 
#in google open that image in new tab and paste the link in the "url " variable

def exp7_chain_code():
    
    img = np.zeros((100, 100), dtype=np.uint8)

    
    img[20:80, 20:80] = 255

   
    contours, _ = cv2.findContours(img,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    
    contour = max(contours, key=len)

    
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

   
    chain_code = []

    for i in range(len(approx) - 1):
        dx = approx[i + 1][0][0] - approx[i][0][0]
        dy = approx[i + 1][0][1] - approx[i][0][1]

       
        if dx > 0 and dy == 0:
            chain_code.append(0)  
        elif dx == 0 and dy > 0:
            chain_code.append(1)  
        elif dx < 0 and dy == 0:
            chain_code.append(2)  
        elif dx == 0 and dy < 0:
            chain_code.append(3)  

   
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    contour_img = np.zeros((100, 100, 3), dtype=np.uint8)

    for p in approx:
        cv2.circle(contour_img, tuple(p[0]), 2, (0, 255, 0), -1)

    plt.imshow(contour_img)
    plt.title("Contour Points")
    plt.axis('off')

    plt.show()

    print("Freeman Chain Code:", chain_code)
exp7_chain_code()
