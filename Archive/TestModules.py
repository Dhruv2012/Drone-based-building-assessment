from Helper import *
import open3d as o3d

# DJI_0166_00008, DJI_0166_00137
img = cv2.imread(r"F:\IIIT-H Work\win_det_heatmaps\rrcServerData\planShape\serverData\LEDNet\save\DJI_0166_400\val\DJI_0166_00008.png")
Get2DCoordsFromSegMask(img)


# Visualize depth map
# rgbFile = "F:\IIIT-H Work\win_det_heatmaps\datasets\IIIT-H Dataset\DJI_Dataset\Bakul_SFM\DJI_0166_400_1\00063.jpg"
# depthFile = "F:\IIIT-H Work\win_det_heatmaps\rrcServerData\Results\depth_maps_Bakul_SFM\depth_maps\00063.jpg.geometric"

rgbFile = "00063.jpg"
depthFile = "00063.jpg.geometric.bin"

pcd, srcPxs = getPointCloud(rgbFile, depthFile)
o3d.visualization.draw_geometries([pcd])
