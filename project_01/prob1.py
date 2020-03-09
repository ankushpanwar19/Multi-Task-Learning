import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import load_data



def discretize_pts(velo_pts, bin_resolution):
    '''
    It returns disretized (binned) points with given
    bin resolution
    '''

    min_val = velo_pts.min()
    max_val = velo_pts.max()

    val_range = np.arange(min_val, max_val, bin_resolution)
    discrete_pts = np.digitize(velo_pts, val_range).astype(int)

    return discrete_pts

## Loading segmentation data
fpath_seg = 'data/segmentation_data.p'
seg_data = load_data.load_data(fpath_seg)

velo_pts = seg_data['velodyne']


## Discretizing points in all three directions
x_disc = discretize_pts(velo_pts[:,0], 0.2)
y_disc = discretize_pts(velo_pts[:,1], 0.2)
z_disc = discretize_pts(velo_pts[:,2], 0.3)


## Grouping points in same bin to max intensity value
disc_array = np.array([x_disc, y_disc, z_disc, velo_pts[:,3]]).T
df = pd.DataFrame(disc_array, columns=['x','y','z','intensity'])

bin_intensity = df.groupby(['x','y','z'])['intensity'].max().reset_index()


## Grouping points with same x and y coordinate to to max intensity value
bin_intensity_xy = bin_intensity.groupby(['x','y'])['intensity'].max().reset_index()


## Creating BEV image 
img = bin_intensity_xy.values
im=np.zeros((int(img[:,0].max()),int(img[:,1].max())))

im[(img[:,0]-1).astype(int),(img[:,1]-1).astype(int)]=img[:,2]

plt.imshow(np.flip(im),cmap='gray')
plt.show()