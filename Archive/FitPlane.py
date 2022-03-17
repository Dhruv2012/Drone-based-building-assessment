from Helper import *
import pyransac3d as pyrsc
import open3d as o3d
from pyntcloud import PyntCloud

from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# Load saved point cloud and visualize it
pcd = o3d.io.read_point_cloud("DJI_0166_00063_3D.ply")
o3d.visualization.draw_geometries([pcd])

# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd.points)
print('xyz_load: ')
print(xyz_load)
print('xyz_load shape: ', xyz_load.shape)

### RANSAC version 1
# plane_model, inliers = pcd.segment_plane(distance_threshold=2, ransac_n=3, num_iterations=1000)
# print('plane model: ', plane_model)
# inlier_cloud = pcd.select_by_index(inliers)
# print('inliers shape:', np.asarray(inlier_cloud.points).shape)

# outlier_cloud = pcd.select_by_index(inliers, invert=True)
# print('outliers shape:', np.asarray(outlier_cloud.points).shape)
# projectedOutliers = Project3DCoordOnPlane(np.asarray(outlier_cloud.points), plane_model)
# projected_cloud = o3d.geometry.PointCloud()
# projected_cloud.points = o3d.utility.Vector3dVector(projectedOutliers)

# inlier_cloud.paint_uniform_color([1, 0, 0])
# outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
# projected_cloud.paint_uniform_color([0, 0, 1])
# o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, projected_cloud])


### RANSAC version 2
# plano1 = pyrsc.Plane()
# print('min points for ransac plane fitting:', 0.75*xyz_load.shape[0])
# best_eq, best_inliers = plano1.fit(xyz_load, 5, minPoints=0.75*xyz_load.shape[0])
# print('best eqn:', best_eq)
# plane = pcd.select_by_index(best_inliers).paint_uniform_color([0, 0, 0])
# obb = plane.get_oriented_bounding_box()
# obb2 = plane.get_axis_aligned_bounding_box()
# obb.color = [0, 0, 1]
# obb2.color = [0, 1, 0]
# not_plane = pcd.select_by_index(best_inliers, invert=True)
# o3d.visualization.draw_geometries([not_plane, plane, obb, obb2])


### RANSAC version 3
# cloud = PyntCloud.from_file("DJI_0166_00063_3D.ply")
# cloud.plot()
# # After fitting plane
# is_floor = cloud.add_scalar_field("plane_fit")
# cloud.plot(use_as_color=is_floor, cmap="cool")


########## Project points on a best fit plane ############
plane_model, inliers = pcd.segment_plane(distance_threshold=0.8, ransac_n=3, num_iterations=1000)
plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
o3d.visualization.draw_geometries([plane_cloud])

# Normal vector to the plane
print(a, b, c, d)
mod = np.abs(np.sqrt(a*a + b*b + c*c))
a, b, c, d = a/mod, b/mod, c/mod, d

points_projected_on_plane = []
for p in np.asarray(pcd.points):
	# Projected point on plane
	p_dis_plane = np.dot(p, np.array([a,b,c])) + d
	p_proj = p - (p_dis_plane)*np.array([a,b,c])
	points_projected_on_plane.append(p_proj)

pcd_proj = o3d.geometry.PointCloud()
pcd_proj.points = o3d.utility.Vector3dVector(points_projected_on_plane)
o3d.visualization.draw_geometries([pcd_proj])

points_projected_on_plane_arr = np.asarray(points_projected_on_plane)

### Fitting curve to plane ###
tck, u = splprep(np.asarray(points_projected_on_plane_arr[:,0:2]).T, u=None, s=0.0, per=1) 
u_new = np.linspace(u.min(), u.max(), 1000)
x_new, y_new = splev(u_new, tck, der=0)

polygon_points = zip(x_new, y_new)

plt.plot(points_projected_on_plane_arr[:,0], points_projected_on_plane_arr[:,1], 'ro')
plt.plot(x_new, y_new, 'b--')
plt.show()

print(PolyArea(x_new, y_new))


# ######## Fit a contour on these points ########
# pcd_proj_arr = np.asarray(pcd_proj.points)
# pcd_proj_arr[:,2] = 0 # Z axis does not matter now
# pcd_proj.points = o3d.utility.Vector3dVector(pcd_proj_arr)
# o3d.visualization.draw_geometries([pcd_proj])
# pcd_proj_arr_2d = pcd_proj_arr[:,:2] # 2D points
# pcd_proj_arr_2d[:,1] += int(abs(pcd_proj_arr_2d[pcd_proj_arr_2d[:,1].argmin(),:][1]+1))	# Positive x-y pixels
# pcd_proj_arr_2d[:,0] += int(abs(pcd_proj_arr_2d[pcd_proj_arr_2d[:,0].argmin(),:][0]+1))
# print(pcd_proj_arr_2d)
# extRight = tuple(pcd_proj_arr_2d[pcd_proj_arr_2d[:,0].argmax(),:])
# extDown = tuple(pcd_proj_arr_2d[pcd_proj_arr_2d[:,1].argmax(),:])
# extTop = tuple(pcd_proj_arr_2d[pcd_proj_arr_2d[:,1].argmin(),:])
# extLeft = tuple(pcd_proj_arr_2d[pcd_proj_arr_2d[:,0].argmin(),:])
# print("Extreme R, D, T, L ",extRight, extDown, extTop, extLeft)

# factor = 10;
# img_width = int(extRight[0]*factor+1)
# img_height = int(extDown[1]*factor+1)
# PlanShape = np.zeros((img_height, img_width), np.uint8)
# print("Image Height: ", img_width, img_height)

# for p in pcd_proj_arr_2d:
# 	# print(int(p[0]*(factor/2)),int(p[1]*(factor/2)))
# 	PlanShape[int(p[1]*(factor)),int(p[0]*(factor))] = 255

# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #Ellipsoid kernel


# for i in range(0,5):
# 	PlanShape = cv2.dilate(PlanShape, kernel, iterations=5)
# 	PlanShape = cv2.erode(PlanShape, kernel, iterations=5)
# PlanShape = cv2.dilate(PlanShape, kernel, iterations=3)
# # PlanShape = cv2.erode(PlanShape, kernel, iterations=4)
# # PlanShape = cv2.medianBlur(PlanShape,3)

# cv2.imshow("PlanShape", PlanShape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






	
