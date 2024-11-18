import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgOut = cv2.GaussianBlur(imgGray,(5, 5), 0)
imgOut13 = cv2.GaussianBlur(imgGray,(13, 13), 0)

nrows = 3
ncols = 3

sobelHorizontal = cv2.Sobel(imgOut, cv2.CV_64F,1,0,ksize = 5)
sobelVertical = cv2.Sobel(imgOut, cv2.CV_64F,0,1,ksize = 5)
sobelSum = sobelHorizontal + sobelVertical

canny = cv2.Canny(imgOut,100,200)

plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
#plt.subplot(nrows, ncols,3),plt.imshow(imgOut, cmap = 'gray')
#plt.title('5x5 blur'), plt.xticks([]), plt.yticks([])

#13x13 gaussian blur
#plt.subplot(nrows, ncols,4),plt.imshow(imgOut13, cmap = 'gray')
#plt.title('13x13 blur'), plt.xticks([]), plt.yticks([])

#edge detection image
plt.subplot(nrows, ncols,3),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('sobel H'), plt.xticks([]), plt.yticks([])

#vertical edge detection
plt.subplot(nrows, ncols,4),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('sobel V'), plt.xticks([]), plt.yticks([])

#summed sobel image
plt.subplot(nrows, ncols,5),plt.imshow(sobelSum, cmap = 'gray')
plt.title('sobel summed'), plt.xticks([]), plt.yticks([])

#canny image
plt.subplot(nrows, ncols,6),plt.imshow(canny, cmap = 'gray')
plt.title('canny image'), plt.xticks([]), plt.yticks([])

#maxVal = np.min(sobelSum)

#maxVal in sobelSum: 10326.0
#minVal in sobelSum: -9792.0
#middle val is 267

for val in sobelSum:
    if val < 267:
        val = -9792
    else if val > 267:
        val = 10326

        


plt.show()


