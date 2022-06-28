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
    
    column_names = ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
    data = pd.read_csv(file_path, sep=' ', names=column_names)
    return data

def GetScale(prev_coords, current_cords):

    #Input:
    #prev_coords = (prev_lat, prev_lon)
    #current_cords = (current_lat, current_lon)

    #Output:
    #scale = distance traveled in meters

    scale = abs(geopy.distance.distance(prev_coords, current_cords).m)
    return scale

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



def GetHomographyData(orb_data,log_data):
    
    transformation_matrices={}

    #Adding reference frame data
    ref_translation = [orb_data['x'][0], orb_data['y'][0], orb_data['z'][0]]
    ref_quat = [orb_data['qw'][0], orb_data['qx'][0], orb_data['qy'][0], orb_data['qz'][0]]
    ref_scale = 1
    ref_transformation_matrix = GetTransformationMatrix(ref_translation, ref_quat, ref_scale)

    transformation_matrices[log_data['time'][0]] = [ref_transformation_matrix, log_data['image_name'][0], log_data['height'][0]]

    for index, i in orb_data.iterrows():


        ind = int(np.where(log_data['time']/1000==i['time'])[0])
        
        if int(ind)!=0:
            
            current_coords = (log_data['latitude'][ind], log_data['longitude'][ind])
            prev_coords = (log_data['latitude'][ind-1], log_data['longitude'][ind-1])
            scale = GetScale(prev_coords, current_coords)

            translation = [orb_data['x'][index], orb_data['y'][index], orb_data['z'][index]]
            quat = [orb_data['qw'][index], orb_data['qx'][index], orb_data['qy'][index], orb_data['qz'][index]]
            transformation_matrix = GetTransformationMatrix(translation, quat, scale)
            
            print(transformation_matrix)
            transformation_matrices[i['time']] = [transformation_matrix, log_data['image_name'][ind], log_data['height'][ind]]
    
    return transformation_matrices