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

def exp5_morphology():
    
    img = np.zeros((100, 400), dtype=np.uint8)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'A I P', (10, 70), font, 2, 255, 5)

    
    noise = np.random.random((100, 400)) > 0.95
    img[noise] = 255

    kernel = np.ones((5, 5), np.uint8)

    
    erosion = cv2.erode(img, kernel, iterations=1)      
    dilation = cv2.dilate(img, kernel, iterations=1)    
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 

    
    images = [img, erosion, dilation, opening, closing]
    titles = ["Original", "Erosion", "Dilation", "Opening", "Closing"]

    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

exp5_morphology()
