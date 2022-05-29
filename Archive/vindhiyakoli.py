
from Helper import *
import open3d as o3d
import glob

def DisplaySelectedContour(image_path, click_list):
    
    # Displays the selected contour in an image
    # click_list: list of points selected in the image
    image = cv2.imread(image_path)
    image1 = image
    for i in range(len(click_list)-1):
        cv2.line(image1, click_list[i], click_list[i+1], (0,0,255), 2)
    
    while True:
        cv2.imshow('Selected Contour', image1)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

def GetAllImages(folder_path):
    image_paths = glob.glob(folder_path + '/*.jpg')
    image_paths.sort()
    return image_paths

def GetAllDepthMaps(folder_path):
    depth_map_paths = glob.glob(folder_path + '/*.bin')
    depth_map_paths = sorted(depth_map_paths)
    return depth_map_paths

def GenerateAlternatePointCloud(images_txt_path, image_path, depth_map_path):
    
    List2D = []    
    List3D = []
    for i in range(0,1600,20):
        for j in range(0,896,20):
            List2D.append((i,j))
    
    image_name = image_path.split('/')[-1]
    depth_map = ReadDepthMap(depth_map_path)
    R, t, _, _ = ReadCameraOrientation(images_txt_path, False, None, image_name)
    drone_k = np.array([[1177.1,0,960],[0,1177.1,540],[0,0,1]]) 
    T_Cam_to_World = getH_Inverse_from_R_t(R, t)
    depth_values = DepthValues(List2D, depth_map)
    List3D = Get3Dfrom2D(List2D, drone_k, R, t, depth_values)
    List3D = np.concatenate(List3D, axis=1).T
    # print(List2D)
    print(List3D.shape)
    return List3D
    
def GetMeshDistance(points3d1, points3d2):

    mean1 = np.mean(points3d1, axis=0)
    mean2 = np.mean(points3d2, axis=0)

    mesh_distance = ((mean1 - mean2)**2).sum()**0.5
    return mesh_distance

def DepthValues(List2D, depth_map):
    depth_values = []
    for p in List2D:
        depth_values.append(depth_map[p[1], p[0]])
    return depth_values

def Perform3Dfrom2D(images_txt_path, image_path, depth_map_path):
    
    image_name = image_path.split('/')[-1]
    depth_map = ReadDepthMap(depth_map_path)
    R, t, _, _ = ReadCameraOrientation(images_txt_path, False, None, image_name)
    drone_k = np.array([[1177.1,0,960],[0,1177.1,540],[0,0,1]]) 
    List2D = SelectPointsInImage(image_path)
    DisplaySelectedContour(image_path, List2D)
    T_Cam_to_World = getH_Inverse_from_R_t(R, t)
    depth_values = DepthValues(List2D, depth_map)
    List3D = Get3Dfrom2D(List2D, drone_k, R, t, depth_values)
    List3D = np.concatenate(List3D, axis=1).T
    return List3D

if __name__ == '__main__':


    scale = 9.02
    images_txt_path = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/images/images.txt' 
    image_folder_path = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/images'
    depth_folder_path = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/stereo/depth_maps'
    # images = GetAllImages(image_folder_path)
    # depthmaps = GetAllDepthMaps(depth_folder_path)
    # List3D = np.array([0,0,0]).reshape(1,3)
    
    # for i, j in zip(images, depthmaps):
    #     points = GenerateAlternatePointCloud(images_txt_path, i, j)
    #     List3D = np.vstack((List3D, points))


    image_path1 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/images/8.jpg'
    image_path2 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/images/25.jpg'
    depth_binary_path1 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/stereo/depth_maps/8.jpg.photometric.bin'
    depth_binary_path2 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/stereo/depth_maps/25.jpg.photometric.bin'

    # List3D1 = GenerateAlternatePointCloud(images_txt_path, image_path1, depth_binary_path1)
    # print(List3D1.shape)
    List3D1 = Perform3Dfrom2D(images_txt_path, image_path1, depth_binary_path1)
    List3D2 = Perform3Dfrom2D(images_txt_path, image_path1, depth_binary_path1)
    mesh_distance1 = GetMeshDistance(List3D1, List3D2)
    
    
    List3D3 = Perform3Dfrom2D(images_txt_path, image_path2, depth_binary_path2)    
    List3D4 = Perform3Dfrom2D(images_txt_path, image_path2, depth_binary_path2)
    mesh_distance2 = GetMeshDistance(List3D3, List3D4)
    
    print(mesh_distance1*scale)
    print(mesh_distance2*scale)
    # ply_file = o3d.io.read_point_cloud('/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/dense/0/fused.ply')
    # points = np.asarray(ply_file.points)
    # original_point_cloud = o3d.geometry.PointCloud()
    # original_point_cloud.points = o3d.utility.Vector3dVector(points)
    # original_point_cloud.paint_uniform_color([0, 1, 0])

    # selected_point_cloud1 = o3d.geometry.PointCloud()
    # selected_point_cloud1.points = o3d.utility.Vector3dVector(List3D1)
    # selected_point_cloud1.paint_uniform_color([0, 0, 1])
    
    # selected_point_cloud2 = o3d.geometry.PointCloud()
    # selected_point_cloud2.points = o3d.utility.Vector3dVector(List3D2)
    # selected_point_cloud2.paint_uniform_color([1, 0, 0])

    # selected_point_cloud3 = o3d.geometry.PointCloud()
    # selected_point_cloud3.points = o3d.utility.Vector3dVector(List3D3)
    # selected_point_cloud3.paint_uniform_color([1, 1, 0])

    # selected_point_cloud4 = o3d.geometry.PointCloud()
    # selected_point_cloud4.points = o3d.utility.Vector3dVector(List3D4)
    # selected_point_cloud4.paint_uniform_color([1, 0, 1])

    # o3d.visualization.draw_geometries([original_point_cloud, selected_point_cloud1, selected_point_cloud2, selected_point_cloud3, selected_point_cloud4],
    #                                   width=1080, height=760, zoom=0.1412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
