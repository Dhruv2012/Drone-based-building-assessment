"""
Projects selected 2D points from image to 3D world.

Procedure:
1. Select points along the edges in the image, for now save it in a numpy array.
2. Read that image's R and t from Images.txt
3. Perform projection to 3D and display using matplotlib or open3D

Later:
1. Save selected coordinates for images in a CSV format.
2. Save their 3D projection in a CSV format.

Possibly: make a combined CSV format, something like this:
NameOfImage || Rotation || Translation || Selected 2D points || Projected 3D points.

Project: Building Inspection using Drones - IIITH
"""

from Helper_Dhruv import * # Later, we can expand this class to be a wrapper around our pipeline.
import open3d as o3d

# datasetPath = "../data/"
# datasetPath = r"F:\IIIT-H Work\win_det_heatmaps\rrcServerData\planShape\serverData\LEDNet\save\DJI_0166_400\val"
datasetPath = r"data"
# ResultsPath = "../Results/"
ResultsPath = "Results/"
imageName = "00063.jpg" # This image covers three intersecting edge very well
imagename = r"\DJI_0166_00063.png"
imageName_1 = "00064.jpg" # This image covers three intersecting edge very well
imagename_1 = r"\DJI_0166_00064.png"
images_txt_path = "images.txt"
# depthmap_dir = r"F:\IIIT-H Work\win_det_heatmaps\rrcServerData\Results\depth_maps_Bakul_SFM\depth_maps"
depthmap_dir = r"C:\Users\thaku\Desktop\IIITH\BuildingInspection\Github\Drone-based-building-assessment\data"

# Data structure to store all information using Pandas dataframe
df = pd.DataFrame({'ImageName', 'R', 't', '2D', '3D'})


# Debugging with one image only
R, t, _ = ReadCameraOrientation(ResultsPath+images_txt_path, False, None, imageName)
print(R, t)
R_1, t_1, _ = ReadCameraOrientation(ResultsPath+images_txt_path, False, None, imageName_1)
print(R_1, t_1)

List2D = Get2DCoordsFromSegMask(cv2.imread(datasetPath + imagename))
List2D_H = MakeHomogeneousCoordinates(List2D)
depth_map = readDepth(os.path.join(depthmap_dir, imageName+".geometric.bin"))
print(List2D)
# print(List2D_H)

List2D_1 = Get2DCoordsFromSegMask(cv2.imread(datasetPath+imagename_1))
List2D_H_1 = MakeHomogeneousCoordinates(List2D_1)
depth_map_1 = readDepth(os.path.join(depthmap_dir, imageName_1+".geometric.bin"))
print(List2D_1)
# print(List2D_H_1)

T_Cam_to_World = getH_Inverse_from_R_t(R, t)
print(T_Cam_to_World)

drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt

# Tranform 2D coordinates to 3D coordinates (don't worry about scale)
d = 100
List3D = Get3Dfrom2D(List2D, drone_k, R, t, depth_map, d)
print('3D Points: ', len(List3D))
List3D_1 = Get3Dfrom2D(List2D_1, drone_k, R_1, t_1, depth_map_1, d)
print('3D Points: ', len(List3D_1))

# Plot 3D points in matplotlib
ax = plt.axes(projection='3d')

for p in List3D:
	ax.scatter(p[2], -p[1], p[0], s=50.0, color='r')
	# plt.pause(0.1)
for p_1 in List3D_1:
	ax.scatter(p_1[2], -p_1[1], p_1[0], s=50.0, color='g')
plt.show()

'''
pcd = o3d.geometry.PointCloud()
List3D = np.squeeze(np.array(List3D), axis=2)
print('3D Points: ', List3D.shape)
pcd.points = o3d.utility.Vector3dVector(List3D)
o3d.io.write_point_cloud("DJI_0166_00063_3D.ply", pcd)

# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("./DJI_0166_00063_3D.ply")
o3d.visualization.draw_geometries([pcd_load])
# print(List3D[:][0])
# print(List3D.shape)
# print(List3D[0][:][0])
'''


### Run over all images and reconstruct global 3D point cloud
globalList3D = ReconstructEntire3DStructure(ResultsPath, datasetPath, depthmap_dir)
print("globalList3D:", len(globalList3D))
print(globalList3D[0])

# Save entire 3D reconstruction
pcd = o3d.geometry.PointCloud()
globalList3D = np.squeeze(np.array(globalList3D), axis=2)
print('3D Points: ', globalList3D.shape)
pcd.points = o3d.utility.Vector3dVector(globalList3D)
o3d.io.write_point_cloud("DJI_0166_400_3D_depthscaled.ply", pcd)

# Plot 3D points in matplotlib
ax = plt.axes(projection='3d')
for p in globalList3D:
	ax.scatter(p[2], -p[1], p[0], s=50.0, color='r')
plt.show()


# Load saved point cloud and visualize it
pcd_load = o3d.io.read_point_cloud("DJI_0166_400_3D_depthscaled.ply")
o3d.visualization.draw_geometries([pcd_load])






