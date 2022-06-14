from posixpath import split
import numpy as np
import pyransac3d as pyrsc
import cv2
import glob
import matplotlib.pyplot as plt
import argparse

class GroundNormal():

    def __init__(self, imagestxt_path, points3dtxt_path, ground_segmented_images_path):

        self.imagestxt_path = imagestxt_path
        self.points3dtxt_path = points3dtxt_path
        self.ground_segmented_images_path = ground_segmented_images_path
        self.ground_normal_coefficients = []


    def Keypoints(self):

        #Input: path of images.txt file generated using colmap
        #Output: A dictionary of keypoints for each image, only the keypoints that were used for the reconstruction are returned
        #Points3d_id for each images is also retrived (for the keypoints that were not a part of reconstruction, the points3d_id was -1)

        file = open(self.imagestxt_path, 'r')
        
        #First 4 lines do not contain camera pose or 3d points information
        file_list = file.readlines()[4:]
        keypoints = {}
        for i in range(0,120,2):
            image_id = file_list[i].split(' ')[0]
            
            file_list[i+1] = file_list[i+1].strip('\n')
            all_points = np.asarray(list(file_list[i+1].split(' '))).reshape(-1,3)
            all_points = all_points.astype(np.float64)
            reconstructed_points_indices = np.where(all_points[:,2]!=-1)
            reconstructed_points = all_points[reconstructed_points_indices,:].reshape(-1,3)

            keypoints[int(image_id)] = reconstructed_points
            
        return keypoints
            
    def GroundKeypoints(self, keypoints, threshold=30):
        
        #Input: keypoints dictionary, path of segmented images, threshold to prevenet selection of keypoints lying on the border of the segmented ground plane
        #Output: Filtered dictionary of keypoints, only the keypoints that lie on the ground plane are returned

        segmented_images = sorted([str(file) for file in glob.glob(self.ground_segmented_images_path+'*.jpg')])
        ground_keypoints = {}
        for i in range(len(segmented_images)):
            image = cv2.imread(segmented_images[i])
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_id = int(split(segmented_images[i])[-1].split('_')[0])
            image_keypoints = keypoints[image_id]
            temp_list = []
            for j in range(len(image_keypoints)):
                if image_gray[int(image_keypoints[j,1]),int(image_keypoints[j,0])] >= threshold:
                    temp_list.append(image_keypoints[j,:])
            ground_keypoints[image_id] = np.asarray(temp_list).reshape(-1,3)

        return ground_keypoints


    def GroundPointIndices(self,keypoints):

        #Input: keypoints dictionary
        #Output: A list of point index for each image, the corresponding 3D points can be found out from points3d.txt using these indices

        point_indices = []

        for key in keypoints:
            point_indices.append(list(keypoints[key][:,2]))

        #Flattening the list
        flatten_list = lambda y:[x for a in y for x in flatten_list(a)] if type(y) is list else [y]
        point_indices = flatten_list(point_indices)

        #Selecting only unique point indices
        point_indices = np.array(point_indices)
        point_indices = np.unique(point_indices).astype(int)

        return list(point_indices)

    def Ground3DPoints(self, point_indices):
        
        #Input: List of point_indices, Points3d.txt file path that contains 3D points generated using colmap
        #Output: Array of 3D points, only the 3D points corresponding to the point_indices are returned

        file = open(self.points3dtxt_path, 'r')
        file_list = file.readlines()[3:]
        
        #Creating an array of all the 3d points
        points = []
        
        for i in range(len(file_list)):
            file_list[i] = file_list[i].strip('\n')
            line_list = file_list[i].split(' ')
            points.append([float(line_list[0]), float(line_list[1]), float(line_list[2]), float(line_list[3])])

        points = np.asarray(points).reshape(len(points),4)
        points = points[points[:, 0].argsort()]
    
        #Selecting only the 3D points corresponding to the point_indices
        ground_points = []
        for i in range(len(points)):
            if int(points[i,0]) in point_indices:
                ground_points.append([points[i,1], points[i,2], points[i,3]])
        
        ground_points = np.asarray(ground_points).reshape(-1,3)
        
        return ground_points

    def GroundNormalCoefficients(self):
            
            #Input: Path of images.txt file generated using colmap, path of segmented ground plane images, path of points3d.txt file generated using colmap
            #Output: Plane parameters (a,b,c,d) of the ground plane

            #Getting keypoints that were involved in 3d reconstruction
            keypoints = self.Keypoints()
            
            #Getting keypoints that lie on the ground plane
            ground_keypoints = self.GroundKeypoints(keypoints)

            #Getting point indices of the ground plane
            ground_keypoints_indices = self.GroundPointIndices(ground_keypoints)

            #Selecting only the 3D points corresponding to the ground_keypoints_indicies
            ground_points = self.Ground3DPoints(ground_keypoints_indices)
            
            #Fitting a plane to the ground points
            plane1 = pyrsc.Plane()
            self.ground_normal_coefficients, _ = plane1.fit(ground_points, 0.01)
            
            return [self.ground_normal_coefficients[0], self.ground_normal_coefficients[1], self.ground_normal_coefficients[2]]

parser = argparse.ArgumentParser("Distance between adjacent buildings")
parser.add_argument("-i", "--imagestxt", help="images.txt file generated from colmap", required=True, type=str)
parser.add_argument("-s", "--segmented", help="path to segmented ground plane images", required=True, type=str)
parser.add_argument("-p", "--points3d", help="points3d.txt file generated from colmap", required=True, type=str)

def main():
    global args
    args = parser.parse_args()
    ground_normal = GroundNormal(args.imagestxt, args.points3d, args.segmented)
    # print(args.imagestxt, args.segmented, args.points3d)

    print(ground_normal.GroundNormalCoefficients())

if __name__ == "__main__":
    main()
