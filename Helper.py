# Helper Class

import cv2
import numpy as np
import pandas as pd

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
		t = (np.array(t)).T
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


def Get3Dfrom2D(List2D, K, M):
	# K is camera intrinsic matrix
	# M is concatenated extrinsic matrix : [R t]
	# NOTE: M_INV is not required if M is the transformation from the camera to world frame already.
	K_inv = np.linalg.inv(np.array(K))
	print(M.shape, K.shape)
	List3D = []
	for p in List2D:
		p3D = M.T @ K_inv @ p
		List3D.append(p3D)
	return List3D





