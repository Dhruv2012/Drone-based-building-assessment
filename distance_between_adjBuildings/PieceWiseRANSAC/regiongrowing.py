from sklearn import neighbors
from sklearn.neighbors import KDTree
import open3d as o3d
import numpy as np

def CloudClustering(point_cloud):
    
    _, indices = point_cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    cloud_down = point_cloud.select_by_index(indices)
    points = np.asarray(cloud_down.points)
    left_cloud = points[points[:,0] < 0]
    right_cloud = points[points[:,0] > 0]

    return left_cloud, right_cloud


def GetCurvatureAndNormal(points, K=30):
    
    tree    = KDTree(points, leaf_size=2)
    _, point_cloud_tree = tree.query(points[:len(points)], k=K)
    
    normals = np.empty_like(points)
    curvatures = np.empty((len(points),1))

    for p in range(len(points)):
        
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

def RegionGrowing(points, theta_threshold = 0.25):

    region = []
    normals, curvatures, point_cloud_tree = GetCurvatureAndNormal(points)
    curvature_threshold = np.percentile(curvatures, 98)
    curvature_order = curvatures[:,0].argsort().tolist()

    while len(curvature_order) > 0:
        
        current_seeds = []
        current_region = []
        minimum_curvature_point = curvature_order[0]
        current_seeds.append(minimum_curvature_point)
        current_region.append(minimum_curvature_point)
        curvature_order.remove(minimum_curvature_point)
        seed_value = 0

        while seed_value < len(current_seeds):
            neighbors = point_cloud_tree[current_seeds[seed_value]]
            for q in range(len(neighbors)):
                current_neighbor = neighbors[q]
                if all([current_neighbor in curvature_order, np.arccos(np.abs(np.dot(normals[current_seeds[seed_value]], normals[current_neighbor]))) < theta_threshold]):
                    current_region.append(current_neighbor)
                    curvature_order.remove(current_neighbor)
                    if curvatures[current_neighbor] < curvature_threshold:
                        current_seeds.append(current_neighbor)
            seed_value += 1
        region.append(current_region)
    
    return region



if __name__ == '__main__':

    # Load point cloud
    point_cloud = o3d.io.read_point_cloud("/home/kushagra/IIIT-H/Distance_Module_Datasets/Parallel Between Buildings/DJI_0232/images/DJI_0232.ply")
    cloud = np.asarray(point_cloud.points)
    print(cloud.shape)
    right_cloud, left_cloud = CloudClustering(point_cloud)
    regions = RegionGrowing(right_cloud)
    print(len(regions))
    region1 = sorted(regions, key=len)
    final_region = []
    for x in region1:
        if len(x) > 50:
            final_region.append(x)
    cloud1 = o3d.geometry.PointCloud()
    cloud1.points = o3d.utility.Vector3dVector(np.array(right_cloud[final_region[-1]]).reshape(-1,3))
    cloud1.paint_uniform_color([1, 0, 0])

    cloud2 = o3d.geometry.PointCloud()
    cloud2.points = o3d.utility.Vector3dVector(np.array(right_cloud[final_region[-2]]).reshape(-1,3))
    cloud2.paint_uniform_color([0, 1, 0])

    cloud3 = o3d.geometry.PointCloud()
    cloud3.points = o3d.utility.Vector3dVector(np.array(right_cloud[final_region[-3]]).reshape(-1,3))
    cloud3.paint_uniform_color([0, 0, 1])

    cloud4 = o3d.geometry.PointCloud()
    cloud4.points = o3d.utility.Vector3dVector(np.array(right_cloud[final_region[-4]]).reshape(-1,3))
    cloud4.paint_uniform_color([0, 1, 1])

    cloud5 = o3d.geometry.PointCloud()
    cloud5.points = o3d.utility.Vector3dVector(np.array(right_cloud[final_region[-5]]).reshape(-1,3))
    cloud5.paint_uniform_color([1, 1, 0])

    cloud6 = o3d.geometry.PointCloud()
    cloud6.points = o3d.utility.Vector3dVector(right_cloud)
    cloud6.paint_uniform_color([1, 0, 1])

    o3d.visualization.draw_geometries([cloud6, cloud1, cloud2, cloud3, cloud4, cloud5])