import numpy as np
import pyransac3d as pyrsac
from utils import *

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






# def FindPointVectors(point, point_normal, point_cloud, K=30):

#     tree = KDTree(point_cloud, leaf_size=2)
#     _, point_cloud_tree = tree.query(point, k=K)
#     point_cloud_tree = point_cloud_tree.tolist()
#     neighbors = point_cloud[point_cloud_tree]
#     projected_neighbors = np.empty_like(neighbors)
#     point_vectors = np.empty_like(neighbors)

#     for j in range(len(neighbors)):
#         projected_neighbors[j] = ProjectPointToPlane(neighbors[j], point_normal, point)
#         point_vectors[j] = projected_neighbors[j] - point

#     return point_vectors, projected_neighbors



