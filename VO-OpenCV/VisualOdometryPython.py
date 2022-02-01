'''
	Visual Odometry Project
	Author: Sarvesh Thakur & Shelly Bagchi
	Date: 3/20/2019
'''


# Including Python Libraries
import os
import cv2
import copy
import random
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
# from alive_progress import alive_bar
from tqdm import tqdm

# Including Helper Libraries
# from ReadCameraModel import *
# from UndistortImage import *

frameList = []
depth = []

drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]])


# Determining Pose Matrix
def poseMatrix(distortOld, distortNew, k):
	kp1, des1 = sift.detectAndCompute(distortOld, None);
	kp2, des2 = sift.detectAndCompute(distortNew, None);

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1, des2, k=2)

	pointsMatched = []
	pointsFrom1 = []
	pointsFrom2 = []

	for i, (s,p) in enumerate(matches):
		if s.distance < 1*p.distance:
			pointsMatched.append(s)
			pointsFrom2.append(kp2[s.trainIdx].pt)
			pointsFrom1.append(kp1[s.queryIdx].pt)

	pointsFrom1 = np.int32(pointsFrom1)
	pointsFrom2 = np.int32(pointsFrom2)

	F, mask = cv2.findFundamentalMat(pointsFrom1, pointsFrom2, cv2.FM_RANSAC)

	pointsFrom1 = pointsFrom1[mask.ravel() == 1]
	pointsFrom2 = pointsFrom2[mask.ravel() == 1]

	E = k.T @ F @ k
	retval, R, t, mask = cv2.recoverPose(E, pointsFrom1, pointsFrom2, k)
	return R, t

# Determining Camera Matrix
def cameraMatrix(file):
	frames = []
	for frame in os.listdir(file):
		frames.append(frame)
	fx, fy, cx, cy, G_camera_frames, LUT = ReadCameraModel('Oxford_dataset/model/')
	K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
	return K, LUT


# Determining Homogenous Matrix
def HomogeneousMatrix(R, t):
    h = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    return h


# Undistorting the Images
def undistortImg(img):
    colorimage = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    undistortedimage = UndistortImage(colorimage, LUT)
    gray = cv2.cvtColor(undistortedimage, cv2.COLOR_BGR2GRAY)
    return gray


# SIFT object
sift = cv2.SIFT_create()
fileName = "data/"

# Get Camera Matrix
# k, LUT = cameraMatrix(fileName)

# Get all frames
for frames in os.listdir(fileName):
	frameList.append(frames)
	print(frames)

# Define Matrices
oldHomogeneousMatrix = np.identity(4);
oldTranslationMatrix = np.array([[0, 0, 0, 1]])
oldTranslationMatrix = oldTranslationMatrix.T

ax = plt.axes(projection='3d')

# Get camera's centre pose
for index in tqdm(range(0, len(frameList)-1)):
	image = cv2.imread("%s%s" % (fileName, frameList[index]), 0) # read image
	# undistortedImage = undistortImg(image);
	undistortedImage = image

	newImage = cv2.imread("%s%s" % (fileName, frameList[index+1]), 0) # read next image
	# undistortedNewImage = undistortImg(newImage);
	undistortedNewImage = newImage

	# Find R and T matrix
	R, T = poseMatrix(undistortedImage, undistortedNewImage, drone_k);

	# 
	newHomogeneousMatrix = HomogeneousMatrix(R, T)
	oldHomogeneousMatrix = oldHomogeneousMatrix @ newHomogeneousMatrix # Matrix Multiplication H1 <= H1 H2
	pose = oldHomogeneousMatrix @ oldTranslationMatrix # P <=T1 H1

	ax.scatter(pose[0][0], pose[1][0], pose[2][0], s=50.0, color='r')
	plt.pause(0.05)
	depth.append([pose[0][0], -pose[2][0]])

plt.show()