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

def exp4_frequency_domain():
   
    from scipy import fftpack

    
    # STEP 1: Download and read the image
   
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    urllib.request.urlretrieve(url, "lena.png")

    # reminder: change your file name
    img = cv2.imread("lena.png", 0)

    
    f = fftpack.fft2(img)

   
    fshift = fftpack.fftshift(f)

   
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  
    d = 30  
    # -------------------- Ideal Low Pass Filter --------------------
    
    ideal_mask = np.zeros((rows, cols), np.uint8)
    ideal_mask[crow-d:crow+d, ccol-d:ccol+d] = 1

    # -------------------- Butterworth Low Pass Filter --------------------
    butter_mask = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow)**2 + (v - ccol)**2)
            butter_mask[u, v] = 1 / (1 + (dist / d)**4)

    # -------------------- Gaussian Low Pass Filter --------------------
    gaussian_mask = np.zeros((rows, cols))
    for u in range(rows):
        for v in range(cols):
            dist = np.sqrt((u - crow)**2 + (v - ccol)**2)
            gaussian_mask[u, v] = np.exp(-(dist**2) / (2 * d**2))

    
    def apply_filter(mask):
        # Multiply frequency image with filter mask
        f_filtered = fshift * mask

        # Shift frequency back
        f_ishift = fftpack.ifftshift(f_filtered)

        # Inverse FFT to get spatial image
        img_back = np.real(fftpack.ifft2(f_ishift))
        return img_back

    img_ideal = apply_filter(ideal_mask)
    img_butter = apply_filter(butter_mask)
    img_gaussian = apply_filter(gaussian_mask)

    
    images = [img, magnitude_spectrum, img_ideal, img_butter, img_gaussian]
    titles = ["Original Image", "Magnitude Spectrum",
              "Ideal LPF", "Butterworth LPF", "Gaussian LPF"]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


exp4_frequency_domain()
