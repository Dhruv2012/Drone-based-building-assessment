from Helper import *
import pyransac3d as pyrsc
import open3d as o3d
from pyntcloud import PyntCloud

# Load saved point cloud and visualize it
pcd = o3d.io.read_point_cloud("DJI_0166_00063_3D.ply")
o3d.visualization.draw_geometries([pcd])

# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd.points)
print('xyz_load: ')
print(xyz_load)
print('xyz_load shape: ', xyz_load.shape)

### RANSAC version 1
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)
inlier_cloud.paint_uniform_color([1, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


### RANSAC version 2
plano1 = pyrsc.Plane()
print('min points for ransac plane fitting:', 0.75*xyz_load.shape[0])
best_eq, best_inliers = plano1.fit(xyz_load, 5, minPoints=0.75*xyz_load.shape[0])
plane = pcd.select_by_index(best_inliers).paint_uniform_color([0, 0, 0])
obb = plane.get_oriented_bounding_box()
obb2 = plane.get_axis_aligned_bounding_box()
obb.color = [0, 0, 1]
obb2.color = [0, 1, 0]
not_plane = pcd.select_by_index(best_inliers, invert=True)
o3d.visualization.draw_geometries([not_plane, plane, obb, obb2])


### RANSAC version 3
cloud = PyntCloud.from_file("DJI_0166_00063_3D.ply")
cloud.plot()
# After fitting plane
is_floor = cloud.add_scalar_field("plane_fit")
cloud.plot(use_as_color=is_floor, cmap="cool")
