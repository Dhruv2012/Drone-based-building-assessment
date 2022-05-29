# Helper Class

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import chain
from PIL import Image
import open3d as o3d
from scipy.interpolate import splprep, splev
import collections
import struct

MODEL = "SIMPLE_RADIAL" #"PINHOLE"

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        print(num_cameras)
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    print(cameras)
    return cameras

def readIntrinsics(path, id):
	if MODEL == "PINHOLE":
		cam_lines = np.array([line[:-1].split(' ') for line in open(path) if not line.startswith('#')])
		ids = cam_lines[:,0].astype(np.int) - 1
		w_h_fx_fy_cx_cy = cam_lines[:,2:8].astype(np.float32)
		intrinsics = np.array([[[fx/h,0,cx/w],[0,fy/h,cy/h],[0,0,1]] for w,h,fx,fy,cx,cy in w_h_fx_fy_cx_cy])
		intrinsics[ids] = intrinsics # reorder by id
		# print(intrinsics)
		return intrinsics[id], intrinsics
	elif MODEL == "SIMPLE_RADIAL":
		cam_lines = np.array([line[:-1].split(' ') for line in open(path) if not line.startswith('#')])
		ids = cam_lines[:,0].astype(np.int) - 1
		w_h_f_cx_cy_ = cam_lines[:,2:8].astype(np.float32)
		intrinsics = np.array([[[f,0,cx],[0,f,cy],[0,0,1]] for w,h,f,cx,cy,_ in w_h_f_cx_cy_])
		intrinsics[ids] = intrinsics # reorder by id
		# print(intrinsics)
		return intrinsics[id], intrinsics


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
		Not optimized for task 2 alone. 
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

	old_H = np.eye(4) # Identity transformation

	# This should not be running everytime.
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
		H = getH_from_R_t(R, t)
		old_H = H@old_H
		# print(R)
		# print(t)
		Rs.append(R)
		ts.append(t)
		ID_Rt[ID] = [R, t, old_H]
		Name_Rt[Name] = [R, t, old_H]
	
	if findAll:
		return Rs, ts
	else:
		if findID is not None:
			return ID_Rt[findID][0], ID_Rt[findID][1], ID_Rt[findID][2], ID_Rt
		else:
			return Name_Rt[findName][0], Name_Rt[findName][1], Name_Rt[findName][2], Name_Rt
			
def getH_Inverse_from_R_t(R, t):
	# assuming R, t are numpy array
    h = np.column_stack((R.T, -R.T@t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    assert h.shape == (4,4)
    return h

def getH_from_R_t(R, t):
	# assuming R, t are numpy array
    h = np.column_stack((R, t))
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

def Get3Dfrom2D(List2D, K, R, t, d, H=None):
	# List2D : n x 2 array of pixel locations in an image
	# K : Intrinsic matrix for camera
	# R : Rotation matrix describing rotation of camera frame
	# 	  w.r.t world frame.
	# t : translation vector describing the translation of camera frame
	# 	  w.r.t world frame
	# [R t] combined is known as the Camera Pose.

	List2D = np.array(List2D)
	List3D = []
	d = np.array(d)
	# t.shape = (3,1)

	# if H is not None:
	# 	R = H[:3,:3]
	# 	t = H[:3,3]
	print(List2D.shape)
	for p, q  in zip(List2D, d):
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
		p3D = cam_world + q * unit_vector
		# print("p3D : \n", p3D)
		List3D.append(p3D)

	return List3D

def Get2DCoordsFromSegMask(img):
	"""
		Returns 2D coordinates from contours of segmented roof-top mask
	"""
	imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray.astype(np.uint8), 10, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# print('No of contours:', len(contours))
	
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
	# print('2D Points Max Contour: {}'.format(List2D.shape))
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

def ReadDepthMap(path):
	min_depth_percentile = 5
	max_depth_percentile = 95

	with open(path, "rb") as fid:
		width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
		fid.seek(0)
		num_delimiter = 0
		byte = fid.read(1)
		while True:
			if byte == b"&":
				num_delimiter += 1
				if num_delimiter >= 3:
					break
			byte = fid.read(1)
		array = np.fromfile(fid, np.float32)
	array = array.reshape((width, height, channels), order="F")

	depth_map = np.transpose(array, (1, 0, 2)).squeeze()

	min_depth, max_depth = np.percentile(depth_map, [min_depth_percentile, max_depth_percentile])
	print(min_depth, max_depth)

	depth_map[depth_map < min_depth] = min_depth
	depth_map[depth_map > max_depth] = max_depth

	return depth_map

def getPointCloud(rgbFile, depthFile):
	thresh = 15.0
	scalingFactor = 1.0
	focalX = 1534.66
	focalY = 1534.66
	centerX = 960.0
	centerY = 540.0

	depth = readDepth(depthFile)
	rgb = Image.open(rgbFile)

	points = []
	colors = []
	srcPxs = []

	for v in range(depth.shape[0]):
		for u in range(depth.shape[1]):
			
			Z = depth[v, u] / scalingFactor
			if Z==0: continue
			if (Z > thresh): continue

			X = (u - centerX) * Z / focalX
			Y = (v - centerY) * Z / focalY
			
			srcPxs.append((u, v))
			points.append((X, Y, Z))
			colors.append(rgb.getpixel((u, v)))

	srcPxs = np.asarray(srcPxs).T
	points = np.asarray(points)
	colors = np.asarray(colors)
	
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors/255)
	
	return pcd, srcPxs

def Get3Dfrom2D_DepthMaps(List2D, K, R, t, last_cam, depth_map, scale=1, debug=False, dataFolder=r"data/img", H=None):
	# List2D : n x 2 array of pixel locations in an image
	# K : Intrinsic matrix for camera
	# R : Rotation matrix describing rotation of camera frame
	# 	  w.r.t world frame.
	# t : translation vector describing the translation of camera frame
	# 	  w.r.t world frame
	# [R t] combined is known as the Camera Pose.
	# depth_map : depth map obtained from bin file corresponding to that image
	
	transform = True
	print("[Helper]:[Get3Dfrom2D_DepthMaps]\n")
	List2D = np.array(List2D)
	List3D = []
	# t.shape = (3,1)
	print('depth_map: ', depth_map.shape)
	print('List2D: ', List2D.shape)
	depth_map_copy = depth_map
	min_depth, max_depth = np.percentile(depth_map, [5, 95])

	if H is not None:
		R_ = H[:3,:3]
		t_ = H[:3,3]
		t_.shape =(3,1)
		# print('Rotation and Translation carried:\n', R_, t_)

	
	# Transform camera origin in World coordinate frame
	cam = last_cam
	cam = np.array([0,0,0]).T; cam.shape = (3,1)
	cam_world =  t + R @ cam
	# cam_world = cam
	last_cam = cam_world # For future calls
	print("cam_world : \n", cam_world)

	for p in List2D:
		# skip if index is out of bound
		if p[1] >= depth_map.shape[0] or p[0] >= depth_map.shape[1]:
			continue

		# Homogeneous pixel coordinate
		p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
		# print("pixel: \n", p)

		# Transform pixel in Camera coordinate frame
		pc = np.linalg.inv(K) @ p
		# print("pc : \n", pc, pc.shape)

		# Transform pixel in World coordinate frame
		pw = t + (R@pc)
		# print("pw : \n", pw, t.shape, R.shape, pc.shape)

		# Find a ray from camera to 3d point
		vector = pw - cam_world
		unit_vector = vector / np.linalg.norm(vector)
		# print("unit_vector : \n", unit_vector)
		
		# Point scaled along this ray
		p3D = cam_world + scale*depth_map[p[1], p[0]] * unit_vector
		if debug:
			depth_map_copy[p[1],p[0]] = max_depth + (max_depth+min_depth)/2

		# print("p3D : \n", p3D)
		List3D.append(p3D)

	# if transform:


	if debug:
		plt.imshow(depth_map_copy)
		plt.savefig(dataFolder + '_depth_map_points.png')
		plt.show()

	return List3D, last_cam, R, t


def visualizeFast(List3D, points=50):
	import random
	if points!=-1:
		delta_sample = random.sample(List3D, points)
	else:
		delta_sample = List3D


	ax = plt.axes(projection='3d')
	for p in delta_sample:
		ax.scatter(p[2], -p[1], p[0], s=50.0, color='r')
		# plt.pause(0.001)
	plt.show()

def readDepth(path, debug=False):
	min_depth_percentile = 5
	max_depth_percentile = 95

	with open(path, "rb") as fid:
		width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
		fid.seek(0)
		num_delimiter = 0
		byte = fid.read(1)
		while True:
			if byte == b"&":
				num_delimiter += 1
				if num_delimiter >= 3:
					break
			byte = fid.read(1)
		array = np.fromfile(fid, np.float32)
	
	# print('width {} height {}'.format(width, height))
	array = array.reshape((width, height, channels), order="F")

	depth_map = np.transpose(array, (1, 0, 2)).squeeze()

	min_depth, max_depth = np.percentile(depth_map, [min_depth_percentile, max_depth_percentile])
	# print(min_depth, max_depth)

	depth_map[depth_map < min_depth] = min_depth
	depth_map[depth_map > max_depth] = max_depth

	if debug:
		plt.imshow(depth_map)
		# plt.savefig(dataFolder + '_depth_map.png')
		plt.show()


	return depth_map


def SavePoints(List3D, Name):
	pcd = o3d.geometry.PointCloud()
	globalList3D = np.squeeze(np.array(List3D), axis=2)
	pcd.points = o3d.utility.Vector3dVector(globalList3D)
	o3d.io.write_point_cloud(Name, pcd)


def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def display_inlier_outlier( cloud , ind ): 
    inlier_cloud = cloud.select_by_index( ind ) 
    outlier_cloud = cloud.select_by_index( ind , invert = True ) # Set to True to save points other than ind
    print("Showing outliers (red) and inliers (gray): ") 
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8]) 
    o3d.visualization.draw_geometries([ inlier_cloud , outlier_cloud ],
                                      width=1080, height=760, zoom=0.1412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def demo_crop_geometry(pcd):
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])
    return pcd

