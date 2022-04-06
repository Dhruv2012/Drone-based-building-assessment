"""
Computes area of roof through SFM point cloud
"""

from Helper import *
import time

debug=True
Find3D = True
FindArea = False
dataset = r"data/";
depthDir = dataset
ResultsPath = "Results/"
CameraPoseFile = "images.txt"
CameraIntrinsicFile = "cameras.txt"
CameraIntrinsicFileBIN = "cameras.bin"

imageName = ["00063.jpg", "00064.jpg",  "00327.jpg", "00357.jpg", "00399.jpg"] # can add more images from the "data/extra" folder # "00249.jpg", "00291.jpg",
segImageName = ["00063.png", "00064.png", "00327.png", "00357.png", "00399.png"] # with corresponding masks in "data/extra" folder # "00249.png", "00291.png",


customList = [[(104, 716), (620, 654), (638, 732), (802, 714)], [(458, 556), (862, 380), (871, 464), (980, 416)]]
Rs = []
ts = []
i = 0;
drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt

if Find3D:
	List3DAll = []
	List3DAll_Depths = []
	last_cam = [0,0,0]

	for imgName, segName in zip(imageName, segImageName):
		R, t, H, _ = ReadCameraOrientation(ResultsPath+CameraPoseFile, False, None, imgName)
		K, _ = readIntrinsics(ResultsPath+CameraIntrinsicFile, int(imgName[:-4]))
		print("Intrinsics: ", K)
		# read_cameras_binary(ResultsPath+CameraIntrinsicFileBIN) # better have a cameras.txt(rather than .bin) from dense reconstruction.

		List2D = Get2DCoordsFromSegMask(cv2.imread(dataset + segName))

		# Scaled with depth map
		depthMap = readDepth(dataset + imgName + ".geometric.bin", debug, dataset+imgName)
		List3D_DepthMaps, _, R, t = Get3Dfrom2D_DepthMaps(List2D, K, R.T, -R.T@t, last_cam, depthMap, 1, debug, dataset+imgName, H)

		SavePoints(List3D_DepthMaps, dataset + imgName[:-4]+"_PC_Depth.ply")
		# if i == 1:
		# 	SavePoints(list3D_T, dataset + imgName[:-4]+"_PC_Depth_Transformed.ply")
		List3DAll_Depths = List3DAll_Depths + List3D_DepthMaps

		# Plot
		# visualizeFast(List3D, 200)
		# visualizeFast(List3D_DepthMaps, -1) # When running custom List2D points
		# visualizeFast(List3D_DepthMaps, 200)
		i += 1

	# SavePoints(List3DAll, dataset + "All_PC.ply")
	SavePoints(List3DAll_Depths, dataset + "All_PC_Depth.ply")

# This section is not well written, Hard coding to remove outliers too.
if FindArea:
	index = 0 #1
	imageName = imageName[index] # Right now, just computes area for single image.
	
	# TODO 1 : Remove outliers in depth ply file, otherwise curve fitting fails!
	# pcd = o3d.io.read_point_cloud(dataset + imageName[:-4]+"_PC_Depth.ply") # Depth
	pcd = o3d.io.read_point_cloud(dataset + "OneSideClosed.ply") # Depth
	points = pick_points(pcd)
	# print(points) # selected indexes = [100, 1310, 1532, 2]
	# points = [100, 1310, 1532, 2]
	pcd_plane = pcd.select_by_index(points)

	# # pcd = o3d.io.read_point_cloud(dataset + imageName[:-4]+"_PC.ply") # Scale
	
	plane_model, inliers = pcd_plane.segment_plane(distance_threshold=0.5, ransac_n=3, num_iterations=12)
	# plane_cloud = pcd.select_by_index(inliers)
	# plane_cloud.paint_uniform_color([1.0, 0, 0])

	[a, b, c, d] = plane_model
	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
	o3d.visualization.draw_geometries([pcd])

	# Normal vector to the plane
	print(a, b, c, d)
	mod = np.abs(np.sqrt(a*a + b*b + c*c))
	a, b, c, d = a/mod, b/mod, c/mod, d

	points_projected_on_plane = []
	for p in np.asarray(pcd.points):
		# Projected point on plane
		p_dis_plane = np.dot(p, np.array([a,b,c])) + d
		p_proj = p - (p_dis_plane)*np.array([a,b,c])
		points_projected_on_plane.append(p_proj)

	pcd_proj = o3d.geometry.PointCloud()
	pcd_proj.points = o3d.utility.Vector3dVector(points_projected_on_plane)
	o3d.visualization.draw_geometries([pcd_proj])

	points_projected_on_plane_arr = np.asarray(points_projected_on_plane)

	# remove outliers
	print("Len Final Points", len(pcd_proj.points))
	points_to_remove = [423,432, 476, 480, 481, 482, 483, 1205, 1205, 1206, 1206, 1207, 1207, 1208, 1209, 1210, 1210, 1211, 1212, 1213, 1214, 1244, 1249, 1251, 1253, 1259, 1260, 1262, 1264, 1265, 1266, 1267,800, 802, 804, 806, 807, 205, 1237, 1238, 473, 474, 482, 483, 485, 503, 503, 499, 495, 486, 486, 486, 481, 481, 472, 472, 472, 475, 481, 484, 486, 492, 499, 503, 495, 484, 1248, 1248, 1248, 1242, 1239, 1235, 1236, 1244, 799, 801, 803, 805, 808, 803, 808, 223, 223, 223, 202, 202, 203, 206, 204, 202, 202, 204, 204, 223, 223, 223]
	points_to_remove.sort()
	points_to_remove = list(set(points_to_remove))
	print("Points to remove: ", points_to_remove)
	finalPoints = pcd_proj.select_by_index(points_to_remove, invert=True)
	
	# n = 5
	# for i in range(0, n):
	# 	removePoints = pick_points(finalPoints)
	# 	finalPoints = finalPoints.select_by_index(removePoints, invert=True)
	cl, ind = finalPoints.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
	finalPoints = finalPoints.select_by_index(ind)
	points_projected_on_plane_arr = np.asarray(finalPoints.points)
	

	# Fitting curve to plane ###
	points_projected_on_plane_arr = points_projected_on_plane_arr[1:][~ (np.sum(np.diff(points_projected_on_plane_arr, axis=0) ** 2, axis=1) < 1e-6)] # Depth
	tck, u = splprep(np.asarray(points_projected_on_plane_arr[:,0:2]).T, u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), 1000)
	x_new, y_new = splev(u_new, tck, der=0)

	polygon_points = zip(x_new, y_new)
	plt.plot(points_projected_on_plane_arr[:,0], points_projected_on_plane_arr[:,1], 'ro')
	plt.plot(x_new, y_new, 'b--')
	plt.show()

	print("Area of the roof: ", PolyArea(x_new, y_new))























#### Extra
	# cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.5)
	# print(ind)
	# out = [20, 40, 80, 120]
	# for o in out:
	# 	ind.remove(o)
	
	# display_inlier_outlier(pcd, ind)
	# print(np.asarray(pcd.points))
	# print(len(ind))
	# print(len(cl.points))
	# display_inlier_outlier(cl, ind)
	# o3d.visualization.draw_geometries([pcd])
	# o3d.visualization.draw_geometries([cl])

