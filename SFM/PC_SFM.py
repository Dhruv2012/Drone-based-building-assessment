import numpy as np
import open3d as o3d

# Load in Point cloud data
# Implement PCA to get eigenvectors = Normals for Planes
# Determine location for ground plane and wall plane

def display_inlier_outlier( cloud , ind ): 
    inlier_cloud = cloud.select_by_index( ind ) 
    outlier_cloud = cloud.select_by_index( ind , invert = True ) # Set to True to save points other than ind
    print("Showing outliers (red) and inliers (gray): ") 
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8]) 
    o3d.visualization.draw_geometries([ inlier_cloud , outlier_cloud ],
                                      width=1080, height=760, zoom=0.1412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

def calculateTotalHeight(cloud, ind):
    inlier_cloud = cloud.select_by_index( ind ) 
    inlierPts = np.array(inlier_cloud.points)
    print(inlierPts.shape)

    maxHeightIndices = np.where(inlierPts[:, 1] == np.amax(inlierPts[:,1]))
    print('maxHeightIndices: ', maxHeightIndices)
    print('maxHeightIndicesDim: ', np.ndim(maxHeightIndices))
    maxHeightX = 0.0
    maxHeightY = 0.0
    maxHeightZ = 0.0

    for i in range(np.ndim(maxHeightIndices) - 1):
        print('maxHeightPt:', inlierPts[maxHeightIndices[i], :])
        maxHeightX = inlierPts[maxHeightIndices[i], 0]
        maxHeightY = inlierPts[maxHeightIndices[i], 1]
        maxHeightZ = inlierPts[maxHeightIndices[i], 2]
    minHeightIndices = np.where(inlierPts[:, 1] == np.amin(inlierPts[:,1])) 
    print('minHeightIndices: ', minHeightIndices)
    print('minHeightIndicesDim: ', np.ndim(minHeightIndices))
    minHeightX = inlierPts[minHeightIndices[0], 0]
    minHeightY = inlierPts[minHeightIndices[0], 1]
    minHeightZ = inlierPts[minHeightIndices[0], 2]

    # for i in range(np.ndim(minHeightIndices) - 1):
    #     print(inlierPts[minHeightIndices[i], :])
    #     if (((inlierPts[:,0] < maxHeightX + 0.01) and (inlierPts[:,0] > maxHeightX - 0.01)) and
    #         ((inlierPts[:,2] < maxHeightZ + 0.01) and (inlierPts[:,2] > maxHeightZ - 0.01))):
    #         minHeightX = inlierPts[minHeightIndices[i], 0]
    #         minHeightY = inlierPts[minHeightIndices[i], 1]
    #         minHeightZ = inlierPts[minHeightIndices[i], 2]
    print('minHeightX: ' + str(minHeightX) + ' minHeightY: ' + str(minHeightY) + ' minHeightZ: ' + str(minHeightZ))
    print('maxHeightX: ' + str(maxHeightX) + ' maxHeightY: ' + str(maxHeightY) + ' maxHeightZ: ' + str(maxHeightZ))
    print('Height in SFM scale is: ' + str(maxHeightY - minHeightY))
    return maxHeightY - minHeightY

def displayPointsOnPlane( cloud , ind ): 
    inlier_cloud = cloud.select_by_index( ind ) 
    
    # inlierPts = np.array(cloud.points)
    # print(inlierPts.shape)
    # maxDepthIndices = np.where(inlierPts[:, 2] == np.amax(inlierPts[:,2]))
    # print('maxDepthIndices: ', maxDepthIndices)
    # print('maxDepthIndicesDim: ', np.ndim(maxDepthIndices))
    # for i in range(np.ndim(maxDepthIndices) - 1):
    #     print(inlierPts[maxDepthIndices[i], :])

    # maxHeightIndices = np.where(inlierPts[:, 1] == np.amax(inlierPts[:,1]))
    # print('maxHeightIndices: ', maxHeightIndices)
    # print('maxHeightIndicesDim: ', np.ndim(maxHeightIndices))
    # maxHeightX = 0.0
    # maxHeightY = 0.0
    # maxHeightZ = 0.0

    # for i in range(np.ndim(maxHeightIndices) - 1):
    #     print(inlierPts[maxHeightIndices[i], :])
    #     maxHeightX = inlierPts[maxHeightIndices[i], 0]
    #     maxHeightY = inlierPts[maxHeightIndices[i], 1]
    #     maxHeightZ = inlierPts[maxHeightIndices[i], 2]
    # minHeightIndices = np.where(inlierPts[:, 1] == np.amin(inlierPts[:,1])) 
    # minHeightX = 0.0
    # minHeightY = 0.0
    # minHeightZ = 0.0

    # for i in range(np.ndim(minHeightIndices) - 1):
    #     print(inlierPts[minHeightIndices[i], :])
    #     if (((inlierPts[:,0] < maxHeightX + 0.01) and (inlierPts[:,0] > maxHeightX - 0.01)) and
    #         ((inlierPts[:,2] < maxHeightZ + 0.01) and (inlierPts[:,2] > maxHeightZ - 0.01))):
    #         minHeightX = inlierPts[minHeightIndices[i], 0]
    #         minHeightY = inlierPts[minHeightIndices[i], 1]
    #         minHeightZ = inlierPts[minHeightIndices[i], 2]
    # print('minHeightX: ' + str(minHeightX) + ' minHeightY: ' + str(minHeightY) + ' minHeightZ: ' + str(minHeightZ))
    # print('maxHeightX: ' + str(maxHeightX) + ' maxHeightY: ' + str(maxHeightY) + ' maxHeightZ: ' + str(maxHeightZ))

    outlier_cloud = cloud.select_by_index( ind , invert = True ) # Set to True to save points other than ind
    print("Showing plane points (red) and other (black): ") 
    outlier_cloud.paint_uniform_color([0, 0, 0])
    inlier_cloud.paint_uniform_color([1,0,0]) 
    o3d.visualization.draw_geometries([ inlier_cloud , outlier_cloud ],
                                      width=1080, height=760, zoom=0.1412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def PCA(cloud):
    data = np.array(cloud.points)
    mean = np.mean(data, axis = 0)
    data_adjust = data - mean # Gaussian distribution with mean zero
    matrix = np.cov(data_adjust.T)
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    sort = eigenValues.argsort()[::-1] # higher to less
    eigenValues = eigenValues[sort]
    eigenVectors = eigenVectors[sort]
    return eigenValues, eigenVectors


def main():

    # Visualize SFM point cloud
    cloud = o3d.io.read_point_cloud("fused.ply") # Read the point cloud
    print("Original Point cloud : Data points : ", cloud)
    #o3d.visualization.draw_geometries([cloud], window_name='Original Cloud', 
    #                                  width=1080, height=760, zoom=0.1412,
    #                                  front=[0.4257, -0.2125, -0.8795],
    #                                  lookat=[2.6172, 2.0475, 1.532],
    #                                  up=[-0.0694, -0.9768, 0.2024]) # Visualize the point cloud     

    # Downsampling for outlier removal
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    #display_inlier_outlier(cloud, ind)
    cloud_dwn = cloud.select_by_index(ind)
    print("Original Point cloud : Data points : ", cloud_dwn)
    #o3d.visualization.draw_geometries([cloud_dwn], window_name='Downsampled Cloud', 
    #                                width=1080, height=760, zoom=0.1412,
    #                                front=[0.4257, -0.2125, -0.8795],
    #                                lookat=[2.6172, 2.0475, 1.532],
    #                                up=[-0.0694, -0.9768, 0.2024]) # Visualize the point cloud     

    # Perform PCA
    w, v = PCA(cloud_dwn)

    # Assuming first eigenvector is the ground plane normal based on data distribution
    normal = v[:,2]
    print(normal)

    #point = np.array([0,0,-80]) # passing through origin
    point = np.mean(cloud_dwn.points, axis=0) # passing through origin
    a, b, c = normal
    d = -(np.dot(normal, point))
    print (a, b, c, d)

    # Ground Plane Visualization
    threshold = 0.9
    pointsOnPlaneIndexes = []
    for index, p in enumerate(cloud.points):
        distance = (np.dot(p, normal) + d) / np.abs(np.sqrt(np.linalg.norm(normal)))
        if (np.abs(distance) < threshold):
            pointsOnPlaneIndexes.append(index)
        #else:
        #    print(distance)

    # Visualize plane points
    calculateTotalHeight(cloud, pointsOnPlaneIndexes)
    displayPointsOnPlane(cloud, pointsOnPlaneIndexes)





if __name__ == "__main__":
    main()