import numpy
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random


class ConditionalRANSAC:
    """
    Implementation of planar RANSAC.
    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.
    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.
    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")
    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, ground_plane_normal=None, thresh=0.05, minPoints=1000, maxIteration=1000):
        """
        Find the best equation for a plane.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param ground_plane_normal: normal vector to ground plane. Used to find planes perpendicular to this
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers
        ---
        """
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                              plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
                      ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if (len(pt_id_inliers) > len(best_inliers)):
                if ground_plane_normal is not None:
                    # print('ground plane normal: %s' % ground_plane_normal)
                    dot_product = np.dot(vecC, ground_plane_normal)
                    if abs(dot_product) <= 0.005:
                        # print('dot product: %s' % dot_product)
                        best_eq = plane_eq
                        best_inliers = pt_id_inliers
                else:
                    best_eq = plane_eq
                    best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq

        return self.equation, self.inliers


def horizontal_distance_using_pca(ply_file):
    cloud = o3d.io.read_point_cloud(ply_file)
    print('Points before statistical outlier removal', np.array(cloud.points).shape[0])
    _, indices = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    cloud_down = cloud.select_by_index(indices)
    # points = np.array(cloud_down.points)
    points = np.array(cloud_down.points)
    print('Points after statistical outlier removal', points.shape[0])

    # Standardizing the data
    point_mean = np.mean(points, axis=0)
    point_std = np.std(points, axis=0)
    new_points = np.divide(points - point_mean, point_std)
    # new_points = points - point_mean
    # Identifying principal components
    covariance_matrix = np.cov(new_points.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Finding the direction in which maximum number of points are present
    order = eigen_values.argsort()[::-1]
    sorted_eigen_values = eigen_values[order]
    sorted_eigen_vectors = eigen_vectors[order]
    print(sorted_eigen_vectors)
    print(eigen_values)

    # Dot product
    print('0 and 1:', np.dot(eigen_vectors[:, 0], (eigen_vectors[:, 1])))
    print('1 and 2:', np.dot(eigen_vectors[:, 2], (eigen_vectors[:, 1])))
    print('0 and 2:', np.dot(eigen_vectors[:, 0], (eigen_vectors[:, 2])))

    # To plot eigen vectors in open3d
    points = [[0, 0, 0], list(sorted_eigen_vectors[:, 0]), list(sorted_eigen_vectors[:, 1]),
              list(sorted_eigen_vectors[:, 2])]
    lines = [[0, 1], [0, 2], [0, 3]]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    colors = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines), )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([line_set])

    # Visualize the point cloud
    # cloud = o3d.geometry.PointCloud(PointCloud=cloud, LineSet=line_set)
    o3d.visualization.draw_geometries([cloud_down, line_set], window_name="Original Cloud",
                                      width=1080, height=760, zoom=0.1412,
                                      front=[0.1, -0.2125, -0.9795],
                                      lookat=[2.6172, 0.0475, 1.532],
                                      up=[-0.0694, -2.9768, 0.2024], )

    # points = [[0,0,0], list(sorted_eigen_vectors[:, 2])]
    # lines = [[0, 1]]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # normal = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
    #                                 lines=o3d.utility.Vector2iVector(lines), )

    # o3d.visualization.draw_geometries([cloud_down , normal ], window_name="Original Cloud",
    #                               width=1080, height=760, zoom=0.1412,
    #                               front=[0.1, -0.2125, -0.9795],
    #                               lookat=[2.6172, 0.0475, 1.532],
    #                               up=[-0.0694, -2.9768, 0.2024],)

    ## clustering point cloud using DBSCAN
    # cluster_pointcloud(cloud_down)
    # leftBuilding, rightBuilding = cluster_dummy(cloud_down)

    ## segment plane RANSAC
    # segment_plane_RANSAC(leftBuilding)
    # segment_plane_RANSAC(rightBuilding)
    # segment_plane_RANSAC(cloud_down)
    return new_points, eigen_values, eigen_vectors


def estimate_verticalPlanes(ply_file, ground_normal):
    cloud = o3d.io.read_point_cloud(ply_file)
    print('Points before voxel downsampling', np.array(cloud.points).shape[0])

    # _, indices = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    # cloud_down = cloud.select_by_index(indices)

    # cloud_down = cloud.voxel_down_sample(voxel_size=0.1)
    cloud_down = cloud
    print('Points after voxel downsampling', np.array(cloud_down.points).shape[0])

    ## clustering point cloud using DBSCAN
    # cluster_pointcloud(cloud_down)
    leftBuilding, rightBuilding = cluster_dummy(cloud_down)

    ## segment plane RANSAC
    leftBuildingPlaneModel, _ = segment_plane_RANSAC(leftBuilding, ground_normal)
    print('leftBuildingPlaneModel: {}'.format(leftBuildingPlaneModel))
    rightBuildingPlaneModel, _ = segment_plane_RANSAC(rightBuilding, ground_normal)
    print('rightBuildingPlaneModel: {}'.format(rightBuildingPlaneModel))
    # segment_plane_RANSAC(cloud_down)

    ## Check if max values in both plane models have same sign else multiply by (-1)
    leftBArr = np.array(leftBuildingPlaneModel)
    rightBArr = np.array(rightBuildingPlaneModel)

    print('absLeftPlaneModel: {}'.format(abs(leftBArr[:3])))
    print('absRightPlaneModel: {}'.format(abs(rightBArr[:3])))
    leftBMaxAbs, leftBIndex = np.max(abs(leftBArr[:3])), np.argmax(abs(leftBArr[:3]))
    rightBMaxAbs, rightBIndex = np.max(abs(rightBArr[:3])), np.argmax(abs(rightBArr[:3]))

    print('leftBMax {} rightBMax {}'.format(leftBMaxAbs, rightBMaxAbs))
    print('After flipping sign')
    if leftBIndex != rightBIndex:
        leftBArr = -1.00 * leftBArr
        print('leftBuildingPlaneModel: {}'.format(leftBArr))
    elif leftBIndex == rightBIndex:
        leftBMax = leftBArr[leftBIndex]
        rightBMax = rightBArr[rightBIndex]
        if (leftBMax > 0 and rightBMax < 0) or (leftBMax < 0 and rightBMax > 0):
            leftBArr = -1.00 * leftBArr
            print('leftBuildingPlaneModel: {}'.format(leftBArr))

    ## distance between 2 planes
    d1 = leftBArr[3]
    d2 = rightBArr[3]

    a, b, c, _ = np.mean([leftBArr, rightBArr], axis=0)
    print('a {} b {} c {}'.format(a, b, c))
    dist_between_buildings = abs((d2 - d1) / (a ** 2 + b ** 2 + c ** 2) ** 0.5)
    print('Dist between 2 buildings is: {} mesh units'.format(dist_between_buildings))
    return dist_between_buildings


def cluster_dummy(cloud):
    points = np.array(cloud.points)
    # leftBuildingIndices = np.where(points[:, 2] < 0)
    # rightBuildingIndices = np.where(points[:, 2] > 0)
    leftBuildingPoints = points[points[:, 0] < 0]
    # print('points shape:', leftBuildingPoints.shape)
    rightBuildingPoints = points[points[:, 0] > 0]

    leftBuilding = o3d.geometry.PointCloud()
    leftBuilding.points = o3d.utility.Vector3dVector(leftBuildingPoints)
    leftBuilding.paint_uniform_color([0, 1, 0])

    rightBuilding = o3d.geometry.PointCloud()
    rightBuilding.points = o3d.utility.Vector3dVector(rightBuildingPoints)
    rightBuilding.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([leftBuilding, rightBuilding],
                                      width=1080, height=760, zoom=0.1412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    return leftBuilding, rightBuilding


def cluster_pointcloud(cloud):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            cloud.cluster_dbscan(eps=0.005, min_points=100, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([cloud],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])


def cluster_pointcloud_KMeans(cloud):
    kmeans = KMeans(n_clusters=2, random_state=0)
    clusters = kmeans.fit_predict(cloud.points)
    print('cluster shape:', kmeans.cluster_centers_.shape)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    centers = kmeans.cluster_centers_.reshape(2, 3, 1)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)


def segment_plane_RANSAC(cloud, ground_normal):
    # Ground Plane Normal: -0.01, 0.62, 0.78
    cRANSAC = ConditionalRANSAC()
    plane_model, inliers = cRANSAC.fit(np.array(cloud.points), ground_plane_normal=ground_normal, thresh=0.01,
                                       minPoints=5000,
                                       maxIteration=500)
    '''
    plane_model, inliers = cloud.segment_plane(distance_threshold=0.01,
                                         ransac_n=3000,
                                         num_iterations=1000)
    '''
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = cloud.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = cloud.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.8,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])

    return plane_model, inliers


def plot_data(eigenvectors, points):
    # Plotting eigenvectors in matplotlib

    eig1 = eigenvectors[:, 0]
    eig2 = eigenvectors[:, 1]
    eig3 = eigenvectors[:, 2]
    origin = [0, 0, 0]
    fig = plt.figure()

    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.quiver(*origin, *eig1, color=['r'])
    ax.quiver(*origin, *eig2, color=['b'])
    ax.quiver(*origin, *eig3, color=['g'])
    # plt.show()

    # Plotting the point cloud in matplotlib
    ax1 = fig.add_subplot(122, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax1.scatter(x, y, z, marker='o')
    plt.show()

def scale_calculation(ply_file):

    #From dataset DJI_0243
    #00001.jpg (Ground plane) -  0.037342 -2.13764 1.73608
    #00099.jpg (15m height) - -0.0706908 0.597046 1.7482
    #00080.jpg (15m height) -  -0.0696913 0.596447 1.76543
    scale = 1

    if ply_file == "DJI_0243.ply":
        diff = abs(-2.13764 - 0.597046)
        scale = 15/diff

    elif ply_file == "DJI_0226.ply":
        points = o3d.io.read_point_cloud(ply_file)
        points_array = numpy.asarray(points.points).T
        s1 = points_array.min(axis=1)[1]
        s2 = -2.85210
        scale = 15/abs(s1-s2)

    # elif ply_file == "DJI_0378.ply":
    #     scale = 2.5
    return scale


if __name__ == "__main__":
    ply_file = "DJI_0378.ply"
    ground_normal = [-0.07346979558651287, -0.09468736980492064, 0.9927922698812185]
    mesh_distance = estimate_verticalPlanes(ply_file, ground_normal)
    # scale = scale_calculation(ply_file)
    # print("Distance between the buildings:", scale*mesh_distance)
    # points, eigen_values, eigen_vectors = horizontal_distance_using_pca(ply_file)
    # plot_data(eigen_vectors, points)
