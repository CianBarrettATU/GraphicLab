import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgOut = cv2.GaussianBlur(imgGray,(5, 5), 0)
imgOut13 = cv2.GaussianBlur(imgGray,(13, 13), 0)

nrows = 2
ncols = 2

plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgOut, cmap = 'gray')
plt.title('5x5 blur'), plt.xticks([]), plt.yticks([])

plt.subplot(nrows, ncols,4),plt.imshow(imgOut13, cmap = 'gray')
plt.title('13x13 blur'), plt.xticks([]), plt.yticks([])
plt.show()


