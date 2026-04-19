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

def exp6_thresholding():
    
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    # reminder: change your file name
    img = cv2.imread("lena.png", 0)

    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    
    _, binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    
    _, trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)

    
    _, tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)

    
    _, otsu = cv2.threshold(img, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    titles = ["Original Image",
              "Binary Threshold",
              "Binary Inverse",
              "Truncate",
              "ToZero",
              "Otsu Threshold"]

    images = [img, binary, binary_inv, trunc, tozero, otsu]

    plt.figure(figsize=(15, 8))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
exp6_thresholding()
