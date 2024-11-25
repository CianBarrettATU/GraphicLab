import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blockSize = 2
apeture_size = 3
k = 0.04

imgHarris = imgGray.copy()

dst = cv2.cornerHarris(imgGray, blockSize, apeture_size, k)

plt.imshow(dst, cmap = "gray")
plt.show()

threshold = 0.5

for i in range(len(dst)):
    for j in range(len(dst[j])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris, (j,i),3,(B,G,R),-1)

plt.imshow(imgHarris, cmap = "gray")
plt.show()
