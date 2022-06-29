from turtle import color
from sklearn.neighbors import KDTree
import numpy as np
import open3d as o3d


def GetCurvatureAndNormal(points, K=30):
    
    point_cloud_tree, _ = GetNearestNeighbors(points, K)
    normals = np.empty_like(points)
    curvatures = np.empty((len(points),1))

    for p in range(len(points)):
        # print(p)
        #For each 3d point, find the K nearest neighbors in the point cloud
        neighbors = points[point_cloud_tree[p]]
        #Compute the covariance matrix of the neighbors
        cov = np.cov(neighbors, rowvar=False)
        #Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #Sort the eigenvalues and eigenvectors in ascending order
        sorted_indices = np.argsort(eigenvalues)
        #Normal vector is the eigenvector corresponding to the smallest eigenvalue
        normal = eigenvectors[:,sorted_indices][:,0]
        #Applying direction correction by taking [0,0,0] as the reference
        reference_point = np.array([0,0,0])
        if normal.dot((reference_point-points[p,:])) > 0:
            normals[p] = normal
        else:
            normals[p] = -normal
        
        #Curvature is the ratio of the smallest eigenvalue to the sum of all eigenvalues
        curvatures[p] = eigenvalues[sorted_indices][0]/np.sum(eigenvalues)
    
    return normals, curvatures, point_cloud_tree

def CloudClustering(point_cloud):
    
    _, indices = point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    cloud_down = point_cloud.select_by_index(indices)
    cloud_down = cloud_down.voxel_down_sample(voxel_size=0.05)
    points = np.asarray(cloud_down.points)
    left_cloud = points[points[:,0] < 0]
    right_cloud = points[points[:,0] > 0]

    return left_cloud, right_cloud

def GetNearestNeighbors(points, K=30):
        
    tree = KDTree(points, leaf_size=2)
    distances, point_cloud_tree = tree.query(points[:len(points)], k=K)
    
    return point_cloud_tree, distances

def VisualizePointCloud(points, colour=[1,0,0]):
        
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(colour)
    o3d.visualization.draw_geometries([point_cloud])

def CleanPointCloud(points, n=50):
        
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.remove_statistical_outlier(nb_neighbors=n, std_ratio=0.1)
    points = np.asarray(point_cloud.points).reshape(-1,3)
    
    return points