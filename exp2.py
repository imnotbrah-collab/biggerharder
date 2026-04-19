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

def exp2_histogram():
    # Download the image
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    #reminder:change your image name
    
    img = cv2.imread("lena.png", 0)

   
    img_eq = cv2.equalizeHist(img)

    
    hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([img_eq], [0], None, [256], [0, 256])

   
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_eq, cmap='gray')
    plt.title("Histogram Equalized Image")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.plot(hist_original)
    plt.title("Original Histogram")

    plt.subplot(2, 2, 4)
    plt.plot(hist_equalized)
    plt.title("Equalized Histogram")

    plt.tight_layout()
    plt.show()

exp2_histogram()
