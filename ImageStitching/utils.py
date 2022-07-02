from math import comb
from cv2 import transform
import numpy as np
import pandas as pd
import geopy.distance

def skew_symmetric(v):

    skew=np.array([[0,  -v[2], v[1]],
                    [ v[2], 0,  -v[0]],
                    [-v[1],v[0], 0]],dtype=np.float64)
    return skew 
    
class Quaternion():
    def __init__(self,w=1.,x=0.,y=0.,z=0.,axis_angle=None,euler=None):
        if euler is None:
            self.w=w
            self.x=x
            self.y=y
            self.z=z

        else:
            euler = np.deg2rad(euler)
            roll=euler[0]
            pitch=euler[1]
            yaw=euler[2]
            
            cr=np.cos(roll*0.5)
            sr=np.sin(roll*0.5)
            cp=np.cos(pitch*0.5)
            sp=np.sin(pitch*0.5)
            cy=np.cos(yaw*0.5)
            sy=np.sin(yaw*0.5)
            
            self.w=cr * cp * cy + sr * sy * sp
            self.x=sr * cp * cy - cr * sp * sy
            self.y=cr * sp * cy + sr * cp * sy
            self.z=cr * cp * sy - sr * sp * cy
 
    def to_mat(self):
        q=np.array([self.x,self.y,self.z]).reshape(3,1)
        i=(self.w**2-np.dot(q.T,q))*np.identity(3)
        j=2*np.dot(q,q.T)
        k=2*self.w*skew_symmetric(q)
        return i+j+k
    def to_numpy(self):
        return np.array([self.w,self.x,self.y,self.z])
    def normalize(self):
        mag=np.linalg.norm([self.w,self.x,self.y,self.z])
        return Quaternion(self.w/mag,self.x/mag,self.y/mag,self.z/mag)
    def quat_mult_right(self,q,out='np'):
        quat=np.array([self.x,self.y,self.z]).reshape(3,1)
        s=np.zeros([4,4])
        s[0,1:]=-quat[:,0]
        s[1:,0]=quat[:,0]
        s[1:,1:]=-skew_symmetric(quat)
        sigma=self.w*np.identity(4)+s
        
        if type(q).__name__=='Quaternion':
            prod_np=np.dot(sigma,q.to_numpy())
        else:
            prod_np=np.dot(sigma,q)
        if out=='np':
            return prod_np
        elif out=='Quaternion':
            prod=Quaternion(prod_np[0],prod_np[1],prod_np[2],prod_np[3])
            return prod
    def quat_mult_left(self,q,out='np'):
        quat=np.array([self.x,self.y,self.z]).reshape(3,1)
        s=np.zeros([4,4])
        s[0,1:]=-quat[:,0]
        s[1:,0]=quat[:,0]
        s[1:,1:]=skew_symmetric(quat)
        sigma=self.w*np.identity(4)+s
        
        if type(q).__name__=='Quaternion':
            prod_np=np.dot(sigma,q.to_numpy())
        else:
            prod_np=np.dot(sigma,q)
        if out=='np':
            return prod_np
        elif out=='Quaternion':
            prod=Quaternion(prod_np[0],prod_np[1],prod_np[2],prod_np[3])
            return prod

def ORBReader(file_path):
    
    column_names = ['time', 'x_orb', 'y_orb', 'z_orb', 'qx_orb', 'qy_orb', 'qz_orb', 'qw_orb']
    data = pd.read_csv(file_path, sep=' ', names=column_names)
    return data

def GetGlobalDistance(prev_coords, current_cords):

    #Input:
    #prev_coords = (prev_lat, prev_lon)
    #current_cords = (current_lat, current_lon)

    #Output:
    #dist = global distance traveled in meters

    dist = abs(geopy.distance.distance(prev_coords, current_cords).m)
    return dist

def GetTransformationMatrix(translation, quaternion, scale):
    
    #Input:
    #translation = (x,y,z)
    #quaternion = (w,x,y,z)
    #Output:
    #transformation_matrix = (4x4)

    quaternion = Quaternion(*quaternion)
    quat_mat = quaternion.to_mat()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3,:3] = quat_mat
    transformation_matrix[:3,3] = translation
    transformation_matrix[0:3,3] = transformation_matrix[0:3,3]*scale

    return transformation_matrix

def Combinelogs(flight_log, orb_log):
    #Converting flight logs time from seconds to miliseconds
    flight_log['time'] = flight_log['time']/1000
    combined_log = pd.merge(flight_log, orb_log, on='time')

    return combined_log

def GetHomographyData(combined_logs):
    
    transformation_matrices={}

    #Adding reference frame data
    ref_translation = [combined_logs['x_orb'][0], combined_logs['y_orb'][0], combined_logs['z_orb'][0]]
    ref_quat = [combined_logs['qw_orb'][0], combined_logs['qx_orb'][0], combined_logs['qy_orb'][0], combined_logs['qz_orb'][0]]
    ref_scale = 1
    ref_transformation_matrix = GetTransformationMatrix(ref_translation, ref_quat, ref_scale)

    transformation_matrices[combined_logs['time'][0]] = [ref_transformation_matrix, combined_logs['image_name'][0], combined_logs['height'][0]]

    for index, i in combined_logs.iterrows():
        
        if index!=0:    
            
            current_coords = (combined_logs['latitude'][index], combined_logs['longitude'][index])
            prev_coords = (combined_logs['latitude'][index-1], combined_logs['longitude'][index-1])
            global_distance = GetGlobalDistance(prev_coords, current_coords)

            prev_translation = [combined_logs['x_orb'][index-1], combined_logs['y_orb'][index-1], combined_logs['z_orb'][index-1]]
            translation = [combined_logs['x_orb'][index], combined_logs['y_orb'][index], combined_logs['z_orb'][index]]
            local_distance = np.linalg.norm(np.array(translation)-np.array(prev_translation))

            scale = global_distance/local_distance
            
            print(combined_logs['time'][index], combined_logs['time'][index-1])
            print(global_distance)
            quat = [combined_logs['qw_orb'][index], combined_logs['qx_orb'][index], combined_logs['qy_orb'][index], combined_logs['qz_orb'][index]]
            
            transformation_matrix = GetTransformationMatrix(translation, quat, scale)
            
            # print(transformation_matrix)
            transformation_matrices[combined_logs['time'][index]] = [transformation_matrix, combined_logs['image_name'][index], combined_logs['height'][index]]
    
    return transformation_matrices


def GetHomographyMatrices(homography_data):
    
    #Returns homography matrices between consecutive images
    homography_matrices={}
    normal = np.array([1,0,0]) #Normal taken along z direction

    keys = list(homography_data)
    keys_temp = keys[:-1]
    for i in keys_temp:
        trans_mat1 = homography_data[i][0]
        trans_mat2 = homography_data[keys[keys.index(i)+1]][0]
        r21 = trans_mat2[:3,:3] @ trans_mat1[:3,:3].T
        t21 = trans_mat2[:3,:3] @ (-trans_mat1[:3,:3] @ trans_mat1[0:3,3]) + trans_mat2[0:3,3]
        h21 = r21 - (t21.reshape(3,1) @ normal.reshape(1,3))/homography_data[i][2]

        homography_matrices[homography_data[keys[keys.index(i)+1]][1]] = h21
    
    return homography_matrices
    