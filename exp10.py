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

def exp10_kmeans():
    
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    # Reminder :change your file name
    img = cv2.imread("lena.png")

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    pixels = img_rgb.reshape((-1, 3))

    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 0.85)

    
    k = 4

    _, labels, centers = cv2.kmeans(pixels,
                                    k,
                                    None,
                                    criteria,
                                    10,
                                    cv2.KMEANS_RANDOM_CENTERS)

   
    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(img_rgb.shape)

  
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title("K-Means Segmented Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(centers.reshape(1, k, 3))
    plt.title("Cluster Colors")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


exp10_kmeans()
