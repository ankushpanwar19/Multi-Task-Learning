import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import load_data

## Loading Detection data
fpath_detect = 'data/detection_data.p'
detect_data = load_data.load_data(fpath_detect)

def box_8points_obj_cam0(obj):
    '''
    It takes the object array (eg:[Car, 1.65, 1.67, 3.64, -0.65, 1.71, 46.70, -1.59]) and return 3D points of the bounding box in cam0 frame
    '''
    #center point of the lower face of the box
    center_pt=np.array(obj[4:7])
    idx=[1,0,2]
    #dimension of the box dim=[width,height,length]
    dim=np.array(obj[1:4])[idx]
    
    # calculating box points using center pt and dimension 
    pts_8=[]
    for h in range(2):

        pts_8.append(center_pt + dim*np.array([-0.5,-h,-0.5]))
        pts_8.append(center_pt + dim*np.array([-0.5,-h,0.5]))
        pts_8.append(center_pt + dim*np.array([0.5,-h,0.5]))
        pts_8.append(center_pt + dim*np.array([0.5,-h,-0.5]))
    pts_8_cam0=np.array(pts_8)
    return pts_8_cam0

def ry_rot_cam0(obj,pts_8_cam0,rot=True):
    if rot==True:
        pts_8_cam0 = pts_8_cam0 - np.array([obj[4:7]])
        ry=np.pi/2+obj[-1]
        R_mat=np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-1*math.sin(ry),0,math.cos(ry)]])

        pts_8_cam0=np.dot(R_mat,pts_8_cam0.T)
        pts_8_cam0=pts_8_cam0.T

        pts_8_cam0 = pts_8_cam0 + np.array([obj[4:7]])

    return(pts_8_cam0)
def img_8points_obj_cam2(pts_8_cam0,P_rec):
    '''
    It takes 3D points of box in cam0 frame and intrinsic projection matrices to Cam 2 and returns image points for the box in cam2 image frame
    '''
    # Taking Homogoneous co-ordinate system
    pts_8_cam0 = np.concatenate((pts_8_cam0, np.ones((pts_8_cam0.shape[0],1))),axis=1)
    pts_8_cam0 = pts_8_cam0.T

    #Transformation to cam2 image frame
    T_cam20=P_rec
    image_pts_homo=np.dot(T_cam20,pts_8_cam0)
    image_pts=image_pts_homo[0:2,:]/image_pts_homo[2]
    image_pts=image_pts.T
    return image_pts

def plot_box_img(img,image_pts,color):
    '''
    It takes image, points of the bounding box and color and returns the image with bounding box drawn on it
    '''
    # Drawing lower rectangle
    pts = np.array(image_pts[0:4], np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts],True,(255,0,0),1)

    # Drawing upper rectangle
    pts = np.array(image_pts[4:9], np.int32)
    pts = pts.reshape((-1,1,2))
    img = cv2.polylines(img,[pts],True,(255,0,0),1)

    # drawing connecting lines from lower to upper rectangle
    pts_int=image_pts.astype(int)
    for i in range(4):

        img=cv2.line(img, pt1=(pts_int[i].item(0),pts_int[i].item(1)), pt2=(pts_int[i+4].item(0),pts_int[i+4].item(1)),color=(255, 0, 0), thickness=1 )
    
    return img

img_rt=np.uint8(detect_data['image_2'])
P_rec=detect_data['P_rect_20']
for i in range(len(detect_data['objects'])):
    print(i)
    obj=detect_data['objects'][i]
    if obj[0] in ['Car','Van','Truck','Pedestrian','Cyclist']:
        print(obj)
        pts_8_cam0=box_8points_obj_cam0(obj)

        pts_8_cam0_rt=ry_rot_cam0(obj,pts_8_cam0,rot=True)

        image_pts_rt=img_8points_obj_cam2(pts_8_cam0_rt,P_rec)

        img_rt=plot_box_img(img_rt,image_pts_rt,(255,0,0))

plt.imshow(img_rt)
plt.show()
