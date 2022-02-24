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

from Helper import * # Later, we can expand this class to be a wrapper around our pipeline.
import open3d as o3d

# datasetPath = "../data/"
datasetPath = r"F:\IIIT-H Work\win_det_heatmaps\rrcServerData\planShape\serverData\LEDNet\save\DJI_0166_400\val"
# ResultsPath = "../Results/"
ResultsPath = "./Results/"
imageName = "00063.jpg" # This image covers three intersecting edge very well
imagename = "\DJI_0166_00063.png"
images_txt_path = "images.txt"


# Data structure to store all information using Pandas dataframe
df = pd.DataFrame({'ImageName', 'R', 't', '2D', '3D'})


# Debugging with one image only
R, t, _ = ReadCameraOrientation(ResultsPath+images_txt_path, False, None, imageName)
print(R, t)

# List2D = SelectPointsInImage(datasetPath+imagename)
List2D = Get2DCoordsFromSegMask(cv2.imread(datasetPath+imagename))
List2D_H = MakeHomogeneousCoordinates(List2D)
print(List2D)
print(List2D_H)

T_Cam_to_World = getH_Inverse_from_R_t(R, t)
print(T_Cam_to_World)

drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt

# Tranform 2D coordinates to 3D coordinates (don't worry about scale)
d = 100
List3D = Get3Dfrom2D(List2D, drone_k, R, t, d)
print('3D Points: ', len(List3D))

# Plot 3D points in matplotlib
ax = plt.axes(projection='3d')

for p in List3D:
	ax.scatter(p[2], -p[1], p[0], s=50.0, color='r')
	# plt.pause(0.1)

plt.show()

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








