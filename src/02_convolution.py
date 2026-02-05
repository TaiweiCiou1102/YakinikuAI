import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR.parent/"source"/"test.jpg"
img = cv2.imread(str(IMG_PATH))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])

edge_image = cv2.filter2D(src=gray_img, ddepth=-1,kernel = kernel)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original Gray')
plt.imshow(gray_img,cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Edge Detection (Convolution)')
plt.imshow(edge_image,cmap='gray')
plt.axis('off')

plt.show()