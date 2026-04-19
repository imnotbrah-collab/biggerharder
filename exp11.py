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

def exp11_bayesian():
    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, confusion_matrix
    import seaborn as sns

   
    np.random.seed(42)

    
    class1 = np.random.multivariate_normal(
        mean=[50, 50, 50],
        cov=np.eye(3) * 20,
        size=100
    )

   
    class2 = np.random.multivariate_normal(
        mean=[150, 150, 150],
        cov=np.eye(3) * 25,
        size=100
    )

    
    labels1 = np.zeros(100)
    labels2 = np.ones(100)

   
    X = np.vstack((class1, class2))
    y = np.hstack((labels1, labels2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


exp11_bayesian()
