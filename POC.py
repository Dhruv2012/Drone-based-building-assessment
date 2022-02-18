import cv2, numpy as np

Pw = [10,20,300]
K = np.array([[1535, 0, 960], [0, 1535, 540], [0, 0, 1]])
# R, t represent transformation from world to camera coordinate frame.
R = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [-1*np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, 0, 1]])
t = np.array([10, 10, 0])
t.shape = (3,1)
print(K)
print(R)
print(t.shape)

Pw.append(1)
Pw = np.array(Pw)
Pw.shape = (4,1)
T = np.hstack((R.T,-R.T@t)) # (3,4)
print("3D point : \n", Pw)
print("Transformation: \n",T)

P_image = K @ T @ Pw
u = P_image[0] / P_image[2]# 923#
v = P_image[1] / P_image[2] #576#
print("2D Point Image Plane: \n", u, v)

K_inv = np.linalg.inv(K)
print("Calib Inverse: \n", K_inv)

# Transform pixel to world coordinate
Pimg_h = np.array([u, v, 1])
Pimg_h.shape = (3,1)
Pimg_in_camera_cf = K_inv @ Pimg_h
print("K_inv @ Pimg_h: ", Pimg_in_camera_cf)

# P_img_world = -R.T@t + R.T @ K_inv @ Pimg_h
# stack overflow approach
P_img_world = t + R @ (Pimg_in_camera_cf)
print("Pixel projected in world frame: \n", P_img_world)

# Transform Camera in world frame
p_cam_origin = np.array([0,0,0])
p_cam_origin.shape = (3,1)
p_cam_origin_world = t + R @ p_cam_origin
print("Cam origin in world frame: \n", p_cam_origin_world)

# Unit vector
vector = P_img_world - p_cam_origin_world
vector = vector / np.linalg.norm(np.array(vector))
print("Unit vector in the direction of the 3D point from camera center: \n", vector)

# Parametrize the vector
ro = p_cam_origin_world
ro.shape = (3,1)
d = 100;
r = ro + d*vector
print("Points along the ray:")
r = ro + d*vector
print(r.squeeze())
# for i in range(0, 315):
#     d = i;
#     r = ro + d*vector
#     print(r.squeeze())