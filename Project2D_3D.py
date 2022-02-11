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

datasetPath = "../data/"
ResultsPath = "../Results/"
imageName = "00027.jpg" # This image covers three intersecting edge very well
images_txt_path = "images.txt"


# Data structure to store all information using Pandas dataframe
df = pd.DataFrame({'ImageName', 'R', 't', '2D', '3D'})


# Debugging with one image only
R, t, _ = ReadCameraOrientation(ResultsPath+images_txt_path, False, None, imageName)
print(R, t)

List2D = SelectPointsInImage(datasetPath+imageName)
List2D_H = MakeHomogeneousCoordinates(List2D)
print(List2D)
print(List2D_H)

T_Cam_to_World = getH_Inverse_from_R_t(R, t)
print(T_Cam_to_World)

drone_k = np.array([[1534.66,0,960],[0,1534.66,540],[0,0,1]]) # later make function to read from cameras.txt

List3D_H = Get3Dfrom2D(List2D_H, drone_k, np.array(T_Cam_to_World))
print(List3D_H)




