import numpy as np

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