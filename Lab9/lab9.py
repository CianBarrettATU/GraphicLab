import cv2
import numpy as np
from matplotlib import pyplot as plt

orbIm = cv2.imread('shrek.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp = orb.detect(orbIm,None)

kp,des = orb.compute(orbIm, kp)

img = cv2.imread('shrek.jpg',)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgHarris = img.copy()

imgShiTomasi = img.copy()

nrows = 3
ncols = 3

blockSize = 2
apeture_size = 3
k = 0.04

dst = cv2.cornerHarris(imgGray, blockSize, apeture_size, k)

threshold = .01

B = 0
G = 0 
R = 255 

for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(B, G, R),-1)

maxCorners = 100 
qualityLevel = 0.01
minDistance = 10

corners = cv2.goodFeaturesToTrack(imgGray,maxCorners,qualityLevel,minDistance)

B = 0
G = 255
R = 0

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi, (int(x), int(y)),3,(B,G,R), -1)


plt.subplot(nrows, ncols, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

# Grayscale Image
plt.subplot(nrows, ncols, 2)
plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale Image')
plt.xticks([]), plt.yticks([])

# Harris Corners
plt.subplot(nrows, ncols, 3)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB))
plt.title('Harris Corners')
plt.xticks([]), plt.yticks([])

# Shi-Tomasi Corners
plt.subplot(nrows, ncols, 4)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corners')
plt.xticks([]), plt.yticks([])

plt.subplot(nrows,ncols, 5)
plt.imshow(cv2.drawKeypoints(orbIm,kp,None, color=(0,255,0), flags=0))
plt.title('orb')
plt.xticks([]), plt.yticks([])

plt.show()
