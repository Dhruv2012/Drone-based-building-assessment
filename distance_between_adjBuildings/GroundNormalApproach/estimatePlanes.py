import numpy as np

def PCA(cloud):
    '''
    Input
    cloud: point cloud from COLMAP

    Output
    eigenValues, eigenVectors 
    '''
    data = np.array(cloud.points)
    mean = np.mean(data, axis = 0)
    data_adjust = data - mean # Gaussian distribution with mean zero
    matrix = np.cov(data_adjust.T)
    eigenValues, eigenVectors = np.linalg.eig(matrix)
    sort = eigenValues.argsort()[::-1] # higher to less
    eigenValues = eigenValues[sort]
    eigenVectors = eigenVectors[sort]
    return eigenValues, eigenVectors

## NOTE: Do outlier removal before fitting the plane
def best_fitting_plane(points, equation=False):
    """ Computes the best fitting plane of the given points
    Parameters
    ----------        
    points: array
        The x,y,z coordinates corresponding to the points from which we want
        to define the best fitting plane. Expected format:
            array([
            [x1,y1,z1],
            ...,
            [xn,yn,zn]])
            
    equation(Optional) : bool
            Set the oputput plane format:
                If True return the a,b,c,d coefficients of the plane.
                If False(Default) return 1 Point and 1 Normal vector.    
    Returns
    -------
    a, b, c, d : float
        The coefficients solving the plane equation.

    or

    point, normal: array
        The plane defined by 1 Point and 1 Normal vector. With format:
        array([Px,Py,Pz]), array([Nx,Ny,Nz])
    """
    w, v = PCA(points)

    #: the normal of the plane is the last eigenvector
    normal = v[:,2]
    
    #: get a point from the plane
    point = np.mean(points, axis=0)

    if equation:
        a, b, c = normal
        d = -(np.dot(normal, point))
        return a, b, c, d
        
    else:
        return point, normal