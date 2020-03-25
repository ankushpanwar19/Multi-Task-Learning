import numpy as np
import pandas as pd 
import os
import load_data
import math
import matplotlib.pyplot as plt
import data_utils
import cv2

fpath_seg = 'data/segmentation_data.p'

seg_data = load_data.load_data(fpath_seg)

velo_pts=seg_data['velodyne'][:,:3]

# finding angle between 3D point vector and x-y plane of velo
dist=np.sqrt(np.square(velo_pts[:,0]) + np.square(velo_pts[:,1]))
angle_deg=np.degrees(np.arctan2(velo_pts[:,2],dist))

# laser id to each 3D point considering uniform FOV distribution  among 64 lasers
laser_id= ((angle_deg-angle_deg.min())/(angle_deg.max()-angle_deg.min())*63).astype(int)+1

# Homogenous Co-ordinates and 
velo_pts = np.concatenate((velo_pts, np.ones((velo_pts.shape[0],1))),axis=1)
velo_pts = velo_pts.T

# Velo to Cam Transformation and filtering -ve points
T_cam2_velo=seg_data['T_cam2_velo']
cam2_points=np.dot(T_cam2_velo,velo_pts)
filter_neg=cam2_points[2]>=0
cam2_points=cam2_points[:,filter_neg]
laser_id_pos=laser_id[filter_neg]

# Cam to Image co-ordinates  and filtering out points outside the image frame
image_pts=np.dot(seg_data['K_cam2'],cam2_points[:3,:])
image_pts=image_pts[0:2,:]/image_pts[2]

cam2_image=np.uint8(seg_data['image_2'])

valid_cam2_pts_mask = (image_pts[0]>=0) & (image_pts[0]<cam2_image.shape[1]) & (image_pts[1]>=0) & (image_pts[1]<cam2_image.shape[0])

image_pts = image_pts[:,valid_cam2_pts_mask]

# color based of laser IDs of filtered image points
laser_im_pts=laser_id_pos[valid_cam2_pts_mask]
color = data_utils.line_color(laser_im_pts)
img = data_utils.print_projection_plt(image_pts, color,cam2_image)

cam2_show = plt.imshow(img)

plt.show()
