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

def exp1_point_processing():
   
   
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

    # rename the file as per your choice and replace the name "lena.png" to your "file_name.png"
    urllib.request.urlretrieve(url, "lena.png")

   #reminder: replace your file name 
    img = cv2.imread("lena.png", 0)
    img_negative = 255 - img
    c = 255 / np.log(1 + np.max(img))
    img_log = c * np.log(img + 1)
    img_log = np.array(img_log, dtype=np.uint8)

   
    gamma = 2.2  # Gamma value
    img_power = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)


    min_val, max_val = 50, 200

    img_stretched = np.uint8(
        np.clip((img - min_val) * (255 / (max_val - min_val)), 0, 255)
    )
    bit_plane_7 = (img >> 7) * 255

    titles = [
        "Original Image",
        "Negative Image",
        "Log Transformation",
        "Power Law Transformation",
        "Contrast Stretching",
        "7th Bit Plane"
    ]

    images = [
        img,
        img_negative,
        img_log,
        img_power,
        img_stretched,
        bit_plane_7
    ]
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()
exp1_point_processing()
