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

def exp3_edge_detection():
    # Download image
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    # reminder : change your file name
    img = cv2.imread("lena.png", 0)

   
   
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)

    
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

   
    sobel_combined = np.uint8(cv2.magnitude(sobelx, sobely))

   
    laplacian = np.uint8(np.absolute(cv2.Laplacian(img, cv2.CV_64F)))

    
    images = [img, sobelx, sobely, sobel_combined, laplacian]
    titles = ["Original", "Sobel X", "Sobel Y", "Sobel Combined", "Laplacian"]

    plt.figure(figsize=(14, 8))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

exp3_edge_detection()
