import numpy as np
from utils import GetNearestNeighbors

def EdgeDetection(region):
    
    #Detects 3D points that lie on an edge for a particular region
    #Input: region - Nx3 numpy array of 3D points
    edge_points = []
    n = 50 #nearest neighbours to be considered
    lamda = 1.5 #threshold for edge detection

    point_cloud_tree , _ = GetNearestNeighbors(region, K=n)
    for i in range(len(region)):
        neighbors = region[point_cloud_tree[i]]
        neighbors = np.delete(neighbors, (0), axis=0)
        centroid = np.mean(neighbors, axis=0)
        closest_neighbour = region[point_cloud_tree[i]][1]
        resolution = np.linalg.norm(region[i]-closest_neighbour)
        # print(resolution)

        if np.linalg.norm(centroid-region[i]) > lamda*resolution:
            edge_points.append(region[i])
    
    return np.asarray(edge_points).reshape(-1,3)