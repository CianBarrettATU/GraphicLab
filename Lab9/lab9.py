import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg',)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgHarris = img.copy()

imgShiTomasi = img.copy()

blockSize = 2
apeture_size = 3
k = 0.04

dst = cv2.cornerHarris(imgGray, blockSize, apeture_size, k)

threshold = 0.5

B = 80
G = 80
R = 80 

for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

maxCorners = np.max(imgHarris)

qualityLevel = 0.01
minDistance = 10

corners = cv2.goodFeaturesToTrack(imgGray,maxCorners,qualityLevel,minDistance)

plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = "gray")
plt.show()
