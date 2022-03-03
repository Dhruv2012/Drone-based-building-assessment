# Helper Class

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import chain

def getR_from_q(q):
	# w, x, y, z
	# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	return np.array([
		[-1 + 2 * q[0] ** 2 + 2 * q[1] ** 2,
		 2 * q[1] * q[2] - 2 * q[0] * q[3],
		 2 * q[3] * q[1] + 2 * q[0] * q[2]],

		[2 * q[1] * q[2] + 2 * q[0] * q[3],
		 -1 + 2 * q[0] ** 2 + 2 * q[2] ** 2,
		 2 * q[2] * q[3] - 2 * q[0] * q[1]],

		[2 * q[3] * q[1] - 2 * q[0] * q[2],
		 2 * q[2] * q[3] + 2 * q[0] * q[1],
		 -1 + 2 * q[0] ** 2 + 2 * q[3] ** 2]])


def ReadCameraOrientation(pathIn, findAll=True, findID=None, findName=None):
	"""
		1. Returns the R, t to transform from world frame to camera frame.
		2. If findAll==false, returns the findID camera R,t
		Not optimized for 2 task alone. 
	"""
	ID_Rt = {} # if only few cam R,t required.
	Name_Rt = {}
	with open(pathIn) as f:
		lines = f.readlines()
	# print(len(lines))

	line_count = 0 # Every odd line needs to be skipped, it has 2D points(not using right now).
	Rs = []
	ts = []
	only_transformation_lines = []

	for index, line in enumerate(lines):
		line = line.strip()

		if not line.startswith('#'):
			line_count = line_count + 1

			if line_count % 2 == 1:
				elements = line.split(" ")
				only_transformation_lines.append(elements)

	# print(only_transformation_lines)
	only_transformation_lines.sort(key=lambda x: int(x[0]))

	for line in only_transformation_lines:
		ID = int(line[0])
		Name = line[9]
		q = []
		for i in range(1,5):
			q.append(float(line[i]))
		t = []
		for j in range(5,8):
			t.append(float(line[j]))
		# print(q, t)

		R = getR_from_q(q)
		R.shape = (3,3)
		t = (np.array(t)).T
		t.shape = (3,1)
		# print(R)
		# print(t)
		Rs.append(R)
		ts.append(t)
		ID_Rt[ID] = [R, t]
		Name_Rt[Name] = [R, t]
	
	if findAll:
		return Rs, ts
	else:
		if findID is not None:
			return ID_Rt[findID][0], ID_Rt[findID][1], ID_Rt
		else:
			return Name_Rt[findName][0], Name_Rt[findName][1], Name_Rt

def getH_Inverse_from_R_t(R, t):
	# assuming R, t are numpy array
    h = np.column_stack((R.T, -R.T@t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    assert h.shape == (4,4)
    return h


		
def PlotOdometry(Rs, ts):
	ax = plt.axes(projection='3d')

	transformation_from_camera_to_world = np.identity(4)
	cameraOrigin = np.array([[0,0,0,1]]) # origin
	cameraOrigin = cameraOrigin.T

	for R,t in tqdm(zip(Rs, ts)):
		# print(R)
		# print(t)
		H = getH_from_R_t(R, t)
		# print(H)
		transformation_from_camera_to_world = H
		camera_pose_in_world_frame = transformation_from_camera_to_world @ cameraOrigin
		ax.scatter(camera_pose_in_world_frame[0][0], camera_pose_in_world_frame[1][0], camera_pose_in_world_frame[2][0], cmap='green')
		plt.pause(0.05)
	plt.show()


def SelectPointsInImage(PathIn, Image=None):
	# Returns the selected points in an image in list
	positions, click_list = [], []

	def callback(event, x, y, flags, param):
		if event == 1:
			click_list.append((x,y))
	cv2.namedWindow('SelectPoints')
	cv2.setMouseCallback('SelectPoints', callback)

	if PathIn is not None:
		Image = cv2.imread(PathIn)	# Don't resize, otherwise scale the points accordingly.

	while True:
		cv2.imshow('SelectPoints', Image)
		k = cv2.waitKey(1)
		if k == 27:
			break
	cv2.destroyAllWindows()

	return click_list


def MakeHomogeneousCoordinates(list_points):
	# when dealing with larger database, function should adapt to numpy ways
	homo_list_points = []
	for point in list_points:
		new_point = []
		for p in point:
			new_point.append(p)
		new_point.append(1)
		homo_list_points.append(tuple(new_point))
	return homo_list_points

def Convert3DH_3D(List3DH):
	List3D = []
	for p in List3DH:
		p_new = (p[0]/p[3], p[1]/p[3],p[2]/p[3])
		List3D.append(p_new)
	return np.array(List3D)

def Get3Dfrom2D(List2D, K, R, t, d=1.75):
	# List2D : n x 2 array of pixel locations in an image
	# K : Intrinsic matrix for camera
	# R : Rotation matrix describing rotation of camera frame
	# 	  w.r.t world frame.
	# t : translation vector describing the translation of camera frame
	# 	  w.r.t world frame
	# [R t] combined is known as the Camera Pose.

	List2D = np.array(List2D)
	List3D = []
	# t.shape = (3,1)

	for p in List2D:
		# Homogeneous pixel coordinate
		p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
		# print("pixel: \n", p)

		# Transform pixel in Camera coordinate frame
		pc = np.linalg.inv(K) @ p
		# print("pc : \n", pc, pc.shape)

		# Transform pixel in World coordinate frame
		pw = t + (R@pc)
		# print("pw : \n", pw, t.shape, R.shape, pc.shape)

		# Transform camera origin in World coordinate frame
		cam = np.array([0,0,0]).T; cam.shape = (3,1)
		cam_world = t + R @ cam
		# print("cam_world : \n", cam_world)

		# Find a ray from camera to 3d point
		vector = pw - cam_world
		unit_vector = vector / np.linalg.norm(vector)
		# print("unit_vector : \n", unit_vector)
		
		# Point scaled along this ray
		p3D = cam_world + d * unit_vector
		# print("p3D : \n", p3D)
		List3D.append(p3D)

	return List3D

def Get2DCoordsFromSegMask(img):
	"""
		Returns 2D coordinates from contours of segmented roof-top mask
	"""
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray.astype(np.uint8), 127, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	print('No of contours:', len(contours))
	
	# cv2.drawContours(im, contours, -1, (0,255,0), 3)
	## Draw max area contour
	
	# c = max(contours, key = cv2.contourArea)
	# cv2.drawContours(img, contours, 0, (0,255,0), 3)
	# cv2.drawContours(img, contours, 1, (255,0,0), 3)
	# c = np.squeeze(c, axis=1)

	sorted_contours = sorted(contours, key=cv2.contourArea)
	maxContour = sorted_contours[len(contours) - 1]
	List2D = np.squeeze(maxContour, axis=1)
	cv2.drawContours(img, [maxContour], 0, (255,0,0), 3)
	# plt.imshow(img)
	# plt.show()
	print('2D Points Max Contour: {}'.format(List2D.shape))
	return list(List2D)

	
def Project3DCoordOnPlane(coords3D, plane_model):
	"""
	coord3D = np array of 3D points whose projection needs to be found
	plane_model = [a, b, c, d] where ax + by + cz + d = 0 plane eqn
	Check this https://www.toppr.com/ask/question/if-the-projection-of-point-pvecp-on-the-plane-vecrcdot-vecnq-is-the-points-svecs/ for projection
	"""
	normal = plane_model[:3]
	d = plane_model[3]
	projected3DList = []

	for i in range(coords3D.shape[0]):
		coord3D = coords3D[i]
		# coord3D.squeeze(axis=1)
		# normal.squeeze(axis=1)
		# print('coord3D {} normal {}', coord3D.shape, normal.shape)
		scalar = ((-d - np.dot(coord3D, normal)) / (np.linalg.norm(normal))**2) 
		projectedCoord3D = coord3D + normal * scalar
		projected3DList.append(projectedCoord3D)
	return np.array(projected3DList)


def ReconstructEntire3DStructure(resultsPath, datasetPath):
	"""
	resultsPath: path to COLMAP results 
	"""
	drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt
	images_txt_path = "images.txt"

	globalList = []

	for fileName in os.listdir(datasetPath):
		print('fileName:', fileName)
		if fileName.endswith(".png"):
			imageName = os.path.splitext(fileName)[0]
			# print('imageName:', imageName)
			# print('imgPath:', datasetPath+fileName)
			List2D = Get2DCoordsFromSegMask(cv2.imread(os.path.join(datasetPath, fileName)))
			# Tranform 2D coordinates to 3D coordinates (don't worry about scale)
			d = 100
			if 'DJI_0166_' in imageName:
				imageName = imageName.replace('DJI_0166_', '')
			try:
				R, t, _ = ReadCameraOrientation(resultsPath+images_txt_path, False, None, imageName+".jpg")
			except KeyError as e:
				break
			List3D = Get3Dfrom2D(List2D, drone_k, R, t, d)
			globalList.append(List3D)

	globalList3D = list(chain.from_iterable(globalList))
	return globalList3D