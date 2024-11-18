import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg',)

nrows = 2
ncols = 1

plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(nrows, ncols,2),plt.imshow(imgGray, cmap = 'gray')
#plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()


