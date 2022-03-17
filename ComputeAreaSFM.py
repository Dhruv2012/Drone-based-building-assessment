"""
Computes area of roof through SFM point cloud
"""

from Helper import *

debug=True
Find3D = True
FindArea = False
dataset = r"data/";
depthDir = dataset
ResultsPath = "Results/"
CameraPoseFile = "images.txt"

imageName = ["00063.jpg", "00399.jpg"] # can add more images from the "data/extra" folder
segImageName = ["00063.png", "00399.png"] # with corresponding masks in "data/extra" folder

drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt

if Find3D:
	List3DAll = []
	List3DAll_Depths = []
	for imgName, segName in zip(imageName, segImageName):
		R, t, _ = ReadCameraOrientation(ResultsPath+CameraPoseFile, False, None, imgName)
		print(R, t)

		List2D = Get2DCoordsFromSegMask(cv2.imread(dataset + segName))
		print(np.array(List2D).shape)

		# Scaled with constant value
		List3D = Get3Dfrom2D(List2D, drone_k, R, t, d=1.75)
		print(np.array(List3D).shape)
		SavePoints(List3D, dataset + imgName[:-4]+"_PC.ply")
		List3DAll = List3DAll + List3D

		# Scaled with depth map
		depthMap = readDepth(dataset + imgName + ".geometric.bin", debug, dataset+imgName)
		List3D_DepthMaps = Get3Dfrom2D_DepthMaps(List2D, drone_k, R, t, depthMap, 1, debug, dataset+imgName)
		print(np.array(List3D_DepthMaps).shape)
		SavePoints(List3D_DepthMaps, dataset + imgName[:-4]+"_PC_Depth.ply")
		List3DAll_Depths = List3DAll_Depths + List3D_DepthMaps

		# Plot
		visualizeFast(List3D, 200)
		visualizeFast(List3D_DepthMaps, 200)

	SavePoints(List3DAll, dataset + "All_PC.ply")
	SavePoints(List3DAll_Depths, dataset + "All_PC_Depth.ply")

if FindArea:
	index = 0 #1
	imageName = imageName[index] # Right now, just computes area for single image.
	
	# TODO 1 : Remove outliers in depth ply file, otherwise curve fitting fails!
	# pcd = o3d.io.read_point_cloud(dataset + imageName[:-4]+"_PC_Depth.ply") # Depth
	pcd = o3d.io.read_point_cloud(dataset + imageName[:-4]+"_PC.ply") # Scale
	plane_model, inliers = pcd.segment_plane(distance_threshold=0.8, ransac_n=3, num_iterations=1500)
	plane_cloud = pcd.select_by_index(inliers)
	plane_cloud.paint_uniform_color([1.0, 0, 0])

	[a, b, c, d] = plane_model
	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
	o3d.visualization.draw_geometries([plane_cloud])

	# Normal vector to the plane
	print(a, b, c, d)
	mod = np.abs(np.sqrt(a*a + b*b + c*c))
	a, b, c, d = a/mod, b/mod, c/mod, d

	points_projected_on_plane = []
	for p in np.asarray(plane_cloud.points):
		# Projected point on plane
		p_dis_plane = np.dot(p, np.array([a,b,c])) + d
		p_proj = p - (p_dis_plane)*np.array([a,b,c])
		points_projected_on_plane.append(p_proj)

	pcd_proj = o3d.geometry.PointCloud()
	pcd_proj.points = o3d.utility.Vector3dVector(points_projected_on_plane)
	o3d.visualization.draw_geometries([pcd_proj])

	points_projected_on_plane_arr = np.asarray(points_projected_on_plane)

	### Fitting curve to plane ###
	# points_projected_on_plane_arr = points_projected_on_plane_arr[1:][~ (np.sum(np.diff(points_projected_on_plane_arr, axis=0) ** 2, axis=1) < 1e-6)] # Depth
	tck, u = splprep(np.asarray(points_projected_on_plane_arr[:,0:2]).T, u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), 1000)
	x_new, y_new = splev(u_new, tck, der=0)

	polygon_points = zip(x_new, y_new)

	plt.plot(points_projected_on_plane_arr[:,0], points_projected_on_plane_arr[:,1], 'ro')
	plt.plot(x_new, y_new, 'b--')
	plt.plot(x_new[0], y_new[0], 'k.')
	# plt.plot(x_new[-1], y_new[-1], 'k.')
	# plt.plot(x_new[-10], y_new[-10], 'k.')
	plt.plot(x_new[100], y_new[100], 'k.')
	plt.show()

	print("Area of the roof: ", PolyArea(x_new, y_new))

