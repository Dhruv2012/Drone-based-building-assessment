import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def horizontal_distance_using_pca(ply_file):
    cloud = o3d.io.read_point_cloud(ply_file)
    # cloud, indices = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    # cloud_down = cloud.select_by_index(indices)
    # print(cloud_down)
    points = np.array(cloud.points)

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
    print(np.dot(eigen_vectors[:, 0], (eigen_vectors[:, 1])))

    # To plot eigen vectors in open3d
    points = [[0, 0, 0], list(eigen_vectors[:, 0]), list(eigen_vectors[:, 1]), list(eigen_vectors[:, 2])]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines), )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([line_set])

    # Visualize the point cloud
    # cloud = o3d.geometry.PointCloud(PointCloud=cloud, LineSet=line_set)
    # o3d.visualization.draw_geometries([cloud], window_name="Original Cloud",
    #                               width=1080, height=760, zoom=0.1412,
    #                               front=[0.1, -0.2125, -0.9795],
    #                               lookat=[2.6172, 0.0475, 1.532],
    #                               up=[-0.0694, -2.9768, 0.2024],)
    return new_points, eigen_values, eigen_vectors


def plot_data(eigenvectors, points):
    # Plotting eigenvectors in matplotlib

    eig1 = eigenvectors[:, 0]
    eig2 = eigenvectors[:, 1]
    eig3 = eigenvectors[:, 2]
    origin = [0, 0, 0]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.quiver(*origin, *eig1, color=['r'])
    ax.quiver(*origin, *eig2, color=['b'])
    ax.quiver(*origin, *eig3, color=['g'])
    plt.show()

    # Plotting the point cloud in matplotlib
    ax1 = fig.add_subplot(111, projection='3d')
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ax1.scatter(x, y, z, marker='o')
    plt.show()


if __name__ == "__main__":
    ply_file = "DJI_0226.ply"
    points, eigen_values, eigen_vectors = horizontal_distance_using_pca(ply_file)
    plot_data(eigen_vectors, points)
