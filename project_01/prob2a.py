import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import load_data
import utils

#%%
fpath_seg = 'data/segmentation_data.p'

seg_data = load_data.load_data(fpath_seg)
cam2_img = seg_data['image_2']


velo_pts = seg_data['velodyne']
velo_pts[:,3] = 1
velo_pts = velo_pts.T

cam2_pts = np.dot(seg_data['T_cam2_velo'],velo_pts)

pos_pts_mask = cam2_pts[2]>0

cam2_pts = cam2_pts[:,pos_pts_mask]
cam2_pts = np.dot(seg_data['K_cam2'], cam2_pts[:3,:])

cam2_img_pts = cam2_pts/cam2_pts[2]
cam2_img_pts = cam2_img_pts.astype(int)


valid_cam2_pts_mask = (cam2_img_pts[0]>=0) & (cam2_img_pts[0]<cam2_img.shape[1]) & (cam2_img_pts[1]>=0) & (cam2_img_pts[1]<cam2_img.shape[0])

cam2_img_pts = cam2_img_pts[:,valid_cam2_pts_mask]

sem_labels = seg_data['sem_label'][pos_pts_mask]
sem_labels = sem_labels[valid_cam2_pts_mask]

color_array = np.empty((sem_labels.shape[0],3))
for idx,i in enumerate(sem_labels[:,0]):
    color_array[idx] = seg_data['color_map'][i][::-1]


cam2_show = plt.imshow(cam2_img)
plt.scatter(x=cam2_img_pts[0,:], y=cam2_img_pts[1,:], c=color_array/255.0, s=1, marker='o')

plt.show()