from regiongrowing import RegionGrowing
from edgedetection import EdgeDetection
from boundrydetection import BoundryDetection
from utils import CloudClustering, VisualizePointCloud, CleanPointCloud
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    
    point_cloud = o3d.io.read_point_cloud("/home/kushagra/IIIT-H/Distance_Module_Datasets/FromTopVideos/DJI_0378/dense/0/fused.ply")
    cloud = np.asarray(point_cloud.points)
    
    left_cloud, right_cloud = CloudClustering(point_cloud)
    
    # VisualizePointCloud(right_cloud, colour =[1,1,0])
    # VisualizePointCloud(left_cloud, colour =[0,1,0])
    # right_cloud_edges = EdgeDetection(right_cloud)
    # left_cloud_edges = EdgeDetection(left_cloud)
    # print(len(left_cloud_edges))
    left_cloud_boundry = BoundryDetection(left_cloud)
    # VisualizePointCloud(left_cloud_edges, colour =[1,0,0])
    # print(left_cloud_boundry.shape)
    print(left_cloud.shape)
    # regions = RegionGrowing(left_cloud_edges)
    # print(len(regions))
    # region1 = sorted(regions, key=len)
    # final_region = []
    # for x in region1:
    #     if len(x) > 100:
    #         final_region.append(x)
    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(left_cloud)
    cloud2.paint_uniform_color([1, 1, 0])
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(left_cloud_boundry)
    cloud1.paint_uniform_color([1, 0, 1])
    # cloud2 = o3d.geometry.PointCloud()
    # cloud2.points = o3d.utility.Vector3dVector(left_cloud)
    # cloud2.paint_uniform_color([1, 1, 0])
    # cloud3 = o3d.geometry.PointCloud()
    # cloud3.points = o3d.utility.Vector3dVector(np.array(left_cloud_edges[final_region[-3]]).reshape(-1,3))
    # cloud3.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([cloud2,cloud1])