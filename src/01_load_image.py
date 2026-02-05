#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
# read image
img = cv2.imread('./source/test.jpg')

if img is None:
    print('Image is not exist. Please check the file path.')
    exit()

red_filter_image = img.copy()
red_filter_image[:,:,0] = 0
red_filter_image[:,:,1] = 0
#red_filter_image[:,:,2] = img[:,:,2]

#%%
plt.imshow(cv2.cvtColor(red_filter_image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()



# %%
