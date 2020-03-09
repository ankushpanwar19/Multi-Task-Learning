import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import data_utils

def undistort_velo_pts(velo_pts, velocity, ang_velocity, do_undistort=False):
    """
    It takes 3D point cloud from velodyne and returns undistorted 3D
    point cloud based on angular and translation velocity
    """
    # intialize undistorted 3D points
    undistorted_velo_pts = velo_pts
    if do_undistort:

        # velo_pts_new =[x,y,1] since points are only distorted in x,y direction 
        velo_pts_new = np.concatenate((velo_pts[:,:2],np.ones((velo_pts.shape[0],1))),axis=1)
        # Angles (in Radian) for each 3D point wrt to velo x-axis (same as Image Taken) (-pi,+pi)
        angles = np.arctan2(velo_pts[:,1],velo_pts[:,0])
        # times_pts = Time difference between Image taken and 3D point scanned
        # time difference could be negative for 3D point taken before Image taken 
        times_pts = -1*angles/(2*np.pi)*0.1
        angle_diff = ang_velocity[-1]*times_pts
        trans_x = velocity[0]*times_pts
        trans_y = velocity[1]*times_pts

        # For creating Transformation matrix (x-y axes only) for each 3D point
        # and calculting undistorted point co-ordinates
        for i in range(velo_pts.shape[0]):
            theta = angle_diff[i]
            Trans_mat = np.array([[np.cos(theta), -np.sin(theta), trans_x[i]],[np.sin(theta), np.cos(theta), trans_y[i]]])

            undistorted_velo_pts[i][:2] = np.dot(Trans_mat,velo_pts_new[i])

    return undistorted_velo_pts

def plot_pts_on_image(velo_pts,cam2_img):
    """
    It takes 3D point cloud from velodyne (distored or undistored) as well as camera image 
    and plots the 3D point on the given image. 
    """
    # creating Homogenous co-ordinates
    velo_pts = np.concatenate((velo_pts, np.ones((velo_pts.shape[0],1))),axis=1)
    velo_pts = velo_pts.T

    # Transformation matrix from velo to camera
    R, T = data_utils.calib_velo2cam('data/problem_4/calib_velo_to_cam.txt')
    T_mat = np.concatenate((R,T), axis=1)

    # 3D point in camera co-ordinates and filtering out the points behind the camera
    cam2_pts = np.dot(T_mat,velo_pts)
    pos_pts_mask = cam2_pts[2]>0
    cam2_pts = cam2_pts[:,pos_pts_mask]

    # Transforming Points to image(pixel) frame from camera co-ordinates 
    # and filtering out the points outside image
    P_mat = data_utils.calib_cam2cam('data/problem_4/calib_cam_to_cam.txt', '02')
    color = data_utils.depth_color(cam2_pts[2])
    cam2_pts = np.dot(P_mat, cam2_pts)
    cam2_img_pts = cam2_pts/cam2_pts[2]
    cam2_img_pts = cam2_img_pts.astype(int)
    valid_cam2_pts_mask = (cam2_img_pts[0]>=0) & (cam2_img_pts[0]<cam2_img.shape[1]) & (cam2_img_pts[1]>=0) & (cam2_img_pts[1]<cam2_img.shape[0])
    cam2_img_pts = cam2_img_pts[:,valid_cam2_pts_mask]

    # color scheme based on point distance 
    img = data_utils.print_projection_plt(cam2_img_pts, color[valid_cam2_pts_mask],cam2_img)
    
    cam2_show = plt.imshow(img)
    plt.show()
    return None

# input image number in 3 digit format as string    
img_no='037'
cam_img = cv2.imread('data/problem_4/image_02/data/0000000'+img_no+'.png')
velo_pts = data_utils.load_from_bin('data/problem_4/velodyne_points/data/0000000'+img_no+'.bin')
velocity = data_utils.load_oxts_velocity('data/problem_4/oxts/data/0000000'+img_no+'.txt')
ang_velocity = data_utils.load_oxts_angular_rate('data/problem_4/oxts/data/0000000'+img_no+'.txt')

u_velo_pts = undistort_velo_pts(velo_pts, velocity, ang_velocity,do_undistort=False)
plot_pts_on_image(u_velo_pts,cam_img)


