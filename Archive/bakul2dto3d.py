from Helper import *
import open3d as o3d
def get_mesh_distance(points3d1, points3d2):

    mean1 = np.mean(points3d1, axis=0)
    mean2 = np.mean(points3d2, axis=0)

    mesh_distance = ((mean1 - mean2)**2).sum()**0.5
    return mesh_distance

if __name__ == '__main__':
    #horizontal edge distance as per google earth = 56.82m
    image_path1 = '/home/kushagra/IIIT-H/2D-3D Testing Bakul 400 Dataset/images/00056.jpg'
    image_path2 = '/home/kushagra/IIIT-H/2D-3D Testing Bakul 400 Dataset/images/00080.jpg'
    # scale = 1.308
    images_txt_path = '/home/kushagra/IIIT-H/2D-3D Testing Bakul 400 Dataset/images/images.txt'

    List3D1 = get3Dpoints(image_path1, images_txt_path)    
    List3D2 = get3Dpoints(image_path2, images_txt_path)

    List3D1 = np.transpose(np.concatenate(List3D1, axis=1))
    List3D2 = np.transpose(np.concatenate(List3D2, axis=1))
    print(List3D1)
    print(List3D2)

    mesh_distance = get_mesh_distance(List3D1, List3D2)
    print(mesh_distance)

