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

def exp8_canny():
    
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    # Reminder change your image name
    img = cv2.imread("lena.png", 0)

    
    # Lower threshold = 50, Upper threshold = 150
    edges_1 = cv2.Canny(img, 50, 150)

    # Moderate threshold values
    edges_2 = cv2.Canny(img, 100, 200)

    # Higher threshold values
    edges_3 = cv2.Canny(img, 150, 250)

    
    titles = [
        "Original Image",
        "Canny (50,150)",
        "Canny (100,200)",
        "Canny (150,250)"
    ]

    images = [img, edges_1, edges_2, edges_3]

    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


exp8_canny()
