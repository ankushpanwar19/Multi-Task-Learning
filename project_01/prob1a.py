#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import load_data
import utils

#%%
fpath_detect = 'data/detection_data.p'
fpath_seg = 'data/segmentation_data.p'

detect_data = load_data.load_data(fpath_detect)
seg_data = load_data.load_data(fpath_seg)

# %%
plt.imshow(seg_data['image_2'])

# %%
utils.describe_dict(seg_data)

# %%
x_min = seg_data['velodyne'][:,0].min()
x_max = seg_data['velodyne'][:,0].max()

y_min = seg_data['velodyne'][:,1].min()
y_max = seg_data['velodyne'][:,1].max()

z_min = seg_data['velodyne'][:,2].min()
z_max = seg_data['velodyne'][:,2].max()

# %%
x_range = np.arange(x_min, x_max, 0.2)
y_range = np.arange(y_min, y_max, 0.2)
z_range = np.arange(z_min, z_max, 0.3)

# %%
x_disc = np.digitize(seg_data['velodyne'][:,0], x_range).astype(int)
y_disc = np.digitize(seg_data['velodyne'][:,1], y_range).astype(int)
z_disc = np.digitize(seg_data['velodyne'][:,2], z_range).astype(int)

# %%
disc_array = np.array([x_disc, y_disc, z_disc, seg_data['velodyne'][:,3]]).T
df = pd.DataFrame(disc_array, columns=['x','y','z','intensity'])

# %%
test = df.groupby(['x','y','z'])['intensity'].max()

# %%
test = test.reset_index()
test_xy = test.groupby(['x','y'])['intensity'].max().reset_index()

# %%
img = test_xy.values

# %%
im=np.zeros((int(img[:,0].max()),int(img[:,1].max())))

# %%
im[(img[:,0]-1).astype(int),(img[:,1]-1).astype(int)]=img[:,2]

# %%
plt.imshow(np.flip(im),cmap='gray')
plt.show()
# %%
cv2.imshow('ImageWindow', np.flip(im))
cv2.waitKey()

# %%
