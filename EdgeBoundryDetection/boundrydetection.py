import numpy as np
from utils import GetCurvatureAndNormal

def ProjectPointsToPlane(points, plane_normal, plane_point):
        
        #Projects a 3D point to a plane
        #Input: point - 3D point to be projected
        #       plane_normal - normal vector of the plane
        #       plane_point - point on the plane
        #Output: projected_point - projected point
        
        [a,b,c] = list(plane_normal.flatten())
        den = a**2 + b**2 + c**2
        [x0,y0,z0] = plane_point
        t = ((a*x0 + b*y0 + c*z0 )/den - np.dot(points,plane_normal.T)/den).reshape(len(points),1)
        projected_points = points + t*plane_normal

        return projected_points

def PlaneTransformation(point_normal, point):

    d = -point_normal.dot(point)
    [a,b,c] = point_normal
    #Returns the points transformed to the xy-plane
    

    #Required translation
    translation = -d/c

    #Required rotation
    costheta = c/((a**2+b**2+c**2)**0.5)
    sintheta = ((a**2+b**2)/(a**2+b**2+c**2))**0.5
    mu1 = b/((a**2+b**2)**0.5)
    mu2 = -a/((a**2+b**2)**0.5)

    rotation = np.zeros((3,3))
    rotation[0,0] = costheta + (mu1**2)*(1-costheta)
    rotation[0,1] = mu1*mu2*(1-costheta)
    rotation[0,2] = mu2*sintheta
    rotation[1,0] = mu1*mu2*(1-costheta)
    rotation[1,1] = costheta + (mu2**2)*(1-costheta)
    rotation[1,2] = -mu1*sintheta
    rotation[2,0] = -mu2*sintheta
    rotation[2,1] = mu1*sintheta
    rotation[2,2] = costheta

    point_normal[2] = point_normal[2] + translation
    new_normal = rotation @ np.asarray(point_normal[:3]).reshape(3,1)

    return new_normal.reshape(1,3)

def CalculateAngles(points):

    #Input: Points projected on xy plane for simplicity

    angles = np.rad2deg(np.arctan2(points[:,0],points[:,1]))

    return angles


def MaxSeparation(angles):

    sorted_angles = angles.copy()
    sorted_angles.sort()
    print(sorted_angles)
    IndividualMaxSeparation = {}
    for i in range(len(sorted_angles)):
        if i==0:
            diff1 = sorted_angles[1] - sorted_angles[0]
            diff2 = 360 - abs(sorted_angles[-1] - sorted_angles[0])
            IndividualMaxSeparation[sorted_angles[0]] = max(abs(diff1),abs(diff2))
        elif i==len(sorted_angles)-1:
            diff1 = sorted_angles[-1] - sorted_angles[-2]
            diff2 = 360 - abs(sorted_angles[0] - sorted_angles[-1])
            IndividualMaxSeparation[sorted_angles[-1]] = max(abs(diff1),abs(diff2))
        else:
            diff1 = sorted_angles[i] - sorted_angles[i-1]
            diff2 = sorted_angles[i+1] - sorted_angles[i]
            IndividualMaxSeparation[sorted_angles[i]] = max(abs(diff1),abs(diff2))
    
    max_separation = max(IndividualMaxSeparation.values())

    return max_separation

def PointVectors(reference_point, neighbors):

    vectors = neighbors - reference_point
    return vectors

def AnglesBetweenVectors(vectors, normal):

    reference = vectors[0]/np.linalg.norm(vectors[0])
    normal = normal/np.linalg.norm(normal)
    angles = []
    for vector in vectors:
        vector = vector/np.linalg.norm(vector)
        angle = np.arctan2(np.dot(np.cross(reference,vector),normal),np.dot(reference,vector))
        angle = np.rad2deg(angle)
        angles.append(angle)
    angles = np.asarray(angles).reshape(1,len(vectors))
    angles = np.where(angles<0,angles+360,angles)
    print(angles)

    return angles.flatten()
def BoundryDetection(region):
    
    boundry_points = []
    n = 50 #nearest neighbours to be considered

    normals, curvatures, point_cloud_tree = GetCurvatureAndNormal(region, K=n)

    for i in range(len(region)):
        point = region[i]
        point_normal = normals[i]
        neighbors = region[point_cloud_tree[i]][1:]
        projected_neighbors = ProjectPointsToPlane(neighbors, point_normal, point)
        print(projected_neighbors)
        projected_neighbor_vectors = PointVectors(point, projected_neighbors)
        print(projected_neighbor_vectors)
        angles = AnglesBetweenVectors(projected_neighbor_vectors, point_normal)
        max_separation = MaxSeparation(angles)
        print(max_separation)
        if max_separation > 120:
            boundry_points.append(point)
    
    return np.asarray(boundry_points).reshape(-1,3)