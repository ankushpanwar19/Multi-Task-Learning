#%%
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import data_utils

#%%
def undistort_velo_pts(velo_pts, velocity, ang_velocity, do_undistort=False):
    undistorted_velo_pts = velo_pts
    if do_undistort:
        velo_pts_new = np.concatenate((velo_pts[:,:2],np.ones((velo_pts.shape[0],1))),axis=1)
        angles = np.arctan2(velo_pts[:,1],velo_pts[:,0])
        times_pts = -1*angles/(2*np.pi)*0.1
        angle_diff = ang_velocity[-1]*times_pts
        trans_x = velocity[0]*times_pts
        trans_y = velocity[1]*times_pts

        for i in range(velo_pts.shape[0]):
            theta = angle_diff[i]
            Trans_mat = np.array([[np.cos(theta), -np.sin(theta), trans_x[i]],[np.sin(theta), np.cos(theta), trans_y[i]]])

            undistorted_velo_pts[i][:2] = np.dot(Trans_mat,velo_pts_new[i])

    return undistorted_velo_pts


#%%
cam2_img = cv2.imread('data/problem_4/image_02/data/0000000310.png')
velo_pts = data_utils.load_from_bin('data/problem_4/velodyne_points/data/0000000310.bin')
velocity = data_utils.load_oxts_velocity('data/problem_4/oxts/data/0000000310.txt')
ang_velocity = data_utils.load_oxts_angular_rate('data/problem_4/oxts/data/0000000310.txt')

velo_pts = undistort_velo_pts(velo_pts, velocity, ang_velocity)
velo_pts = np.concatenate((velo_pts, np.ones((velo_pts.shape[0],1))),axis=1)
velo_pts = velo_pts.T




# %%
R, T = data_utils.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
T_mat = np.concatenate((R,T), axis=1)

# %%
cam2_pts = np.dot(T_mat,velo_pts)

#%%
pos_pts_mask = cam2_pts[2]>0

cam2_pts = cam2_pts[:,pos_pts_mask]

P_mat = data_utils.calib_cam2cam('data/problem_4/calib_cam_to_cam.txt', '02')

#%%
color = data_utils.depth_color(cam2_pts[2])

#%%
cam2_pts = np.dot(P_mat, cam2_pts)

#%%
cam2_img_pts = cam2_pts/cam2_pts[2]
cam2_img_pts = cam2_img_pts.astype(int)

#%%
valid_cam2_pts_mask = (cam2_img_pts[0]>=0) & (cam2_img_pts[0]<cam2_img.shape[1]) & (cam2_img_pts[1]>=0) & (cam2_img_pts[1]<cam2_img.shape[0])

cam2_img_pts = cam2_img_pts[:,valid_cam2_pts_mask]

#%%
img = data_utils.print_projection_plt(cam2_img_pts, color[valid_cam2_pts_mask],cam2_img)

# %%
cam2_show = plt.imshow(img)

plt.show()

# %%
