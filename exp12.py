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

def exp12_svm():
   
    from sklearn.svm import SVC
    from sklearn.datasets import make_blobs
    from sklearn.metrics import accuracy_score

   
    X, y = make_blobs(n_samples=300,
                      centers=2,
                      cluster_std=2,
                      random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    kernels = ['linear', 'rbf', 'poly']

    plt.figure(figsize=(15, 5))

    for i, kernel in enumerate(kernels):
       
        clf = SVC(kernel=kernel)

        
        clf.fit(X_train, y_train)

        
        y_pred = clf.predict(X_test)

        
        accuracy = accuracy_score(y_test, y_pred)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.subplot(1, 3, i + 1)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title(f"Kernel: {kernel}\nAccuracy: {accuracy:.2f}")

    plt.tight_layout()
    plt.show()


exp12_svm()
