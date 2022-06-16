# TODO: Do parameter tuning for right buildings
from types import new_class
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from visualization import *

class ConditionalRANSAC:
    """
    Implementation of planar RANSAC.
    Class for Plane object, which finds the equation of a infinite plane using RANSAC algorithim.
    Call `fit(.)` to randomly take 3 points of pointcloud to verify inliers based on a threshold.
    ![Plane](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/plano.gif "Plane")
    ---
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, ground_plane_normal = None  , thresh=0.05, minPoints=100, maxIteration=1000):
        """
        Find the best equation for a plane.
        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param ground_plane_normal: normal vector to ground plane. Used to find planes perpendicular to this
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers
        ---
        """
        n_points = pts.shape[0]
        best_eq = [0.0,0.0,0.0,0.0]
        best_inliers = []
        
        for it in range(maxIteration):

            # Samples 3 random points
            id_samples = random.sample(range(0, n_points), 3)
            pt_samples = pts[id_samples]

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1

            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
            # We have to use a point to find k
            vecC = vecC / np.linalg.norm(vecC)
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if (len(pt_id_inliers) > len(best_inliers)):
                if ground_plane_normal is not None:
                    # print('ground plane normal: %s' % ground_plane_normal)
                    # dot_product = np.dot(vecC, ground_plane_normal)
                    cross_product=np.cross(vecC,ground_plane_normal)
                    # if abs(np.linalg.norm(cross_product)) <= 0.005:
                        # print('dot product: %s' % dot_product)
                    best_eq = plane_eq
                    best_inliers = pt_id_inliers
                else:
                    best_eq = plane_eq
                    best_inliers = pt_id_inliers
            self.inliers = best_inliers
            self.equation = best_eq
        return self.equation, self.inliers

class AdjacentRoofDistance:
    """
    Class containing all the functions required to segment out roof-top from pointcloud
    and calculate distance.
    """
    def __init__(self,ply_file,scale):
        self.cloud=None
        self.ply_file=ply_file
        self.scale=scale
        self.Leftmodels=[];self.Leftinliers=[];self.Leftclouds = []
        self.Rightmodels=[];self.Rightinliers=[];self.Rightclouds = []
    
    def segment_plane_RANSAC(self,cloud):
        cRANSAC = ConditionalRANSAC()
        plane_model, inliers = cRANSAC.fit(np.array(cloud.points), ground_plane_normal = [0.00, 0.00, 1.00], thresh=0.25,
                                        minPoints=100,
                                        maxIteration=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = cloud.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = cloud.select_by_index(inliers, invert=True)

        # TODO: Add visualization

        return plane_model,inliers,inlier_cloud
    
    def ClusterAlongXaxis(self,cloud):
        points = np.array(self.cloud.points)

        #CLUSTERING ALONG X-AXIS
        newpoints=points[points[:,1]>-2.5]
        newpoints=newpoints[newpoints[:,1]<4.5]
        leftBuildingPoints=newpoints[newpoints[:,0]<-1.0]
        rightBuildingPoints=newpoints[newpoints[:,0]>-1.0]

        leftBuilding = o3d.geometry.PointCloud()
        leftBuilding.points = o3d.utility.Vector3dVector(leftBuildingPoints)
        leftBuilding.paint_uniform_color([0, 1, 0])

        rightBuilding = o3d.geometry.PointCloud()
        rightBuilding.points = o3d.utility.Vector3dVector(rightBuildingPoints)
        rightBuilding.paint_uniform_color([0,0,1])

        # TODO: Add visualization

        return leftBuilding, rightBuilding
    
    def find_dist(self,l1,l2):
        dist = np.sqrt(((l1[0]-l2[0])**2 + (l1[1]-l1[1])**2 + (l1[2]-l2[2])**2))
        return dist 
    
    def getBuildings(self,cloud,k,l,a):
        print("cloud",cloud.shape[0])
        model=[]
        modelinlier=[]
        modelcloud = []
        for i in range(k):
            newcloud = cloud[(cloud[:,0]<(l+(i+1)*a))]
            newcloud = newcloud[(newcloud[:,0]>(l+i*a))] ##Correct in original code as well(Done.)
            print("vals =",(l+i*a), (l+(i+1)*a) )
            print("Here->",(l+i*a), newcloud.shape[0])
            new_cloud = o3d.geometry.PointCloud()
            new_cloud.points = o3d.utility.Vector3dVector(newcloud)
            new_cloud.paint_uniform_color([0,0,1])

            p_model, inlier, cloud1 = self.segment_plane_RANSAC(new_cloud)
            model.append(p_model)
            modelinlier.append(inlier)
            modelcloud.append(cloud1)
        return model,modelinlier,modelcloud

    def getRoof(self,inliers,k):
        index=0
        max = inliers[0].shape[0]
        for i in range(k//2):
            if(inliers[i].shape[0]>max):
                max = inliers[i].shape[0]
                index = i
        return index
    
    def selectPoints(self,cloud,m,yl,ya,offset,index,str):
        listtruepose=[]
        for i in range(m):
        
            y= yl+ya*i
            mypoints=np.array(cloud[index].points)
            truepoints=np.array(cloud[index].points)
            mypoints = mypoints[mypoints[:,1]>(y-offset)]
            mypoints = mypoints[mypoints[:,1]<(y+offset)]
            mymin = mypoints[0,0]
            posmin = 0

            for i in range(mypoints.shape[0]):
                if str == "Left":
                    if(mypoints[i,0]>mymin):
                        mymin=mypoints[i,0]
                        posmin=i
                else:
                    if(mypoints[i,0]<mymin):
                        mymin=mypoints[i,0]
                        posmin=i

            truepos=0
            for i in range(truepoints.shape[0]):

                if((truepoints[i,0]==mypoints[posmin,0]) and (truepoints[i,1]==mypoints[posmin,1]) and (truepoints[i,2]==mypoints[posmin,2])):
                    truepos=i
                    break
            listtruepose.append(truepos)
        return listtruepose

          
    def EstimateDistanceAdjacentBuildings(self):
        # TODO: Need to modularize these parameters
        k=12;l=-4.0;h=-1.0;a=(h-l)/k
        k2=10;l2=2.0;h2=4.5;a2=(h2-l2)/k2

        self.cloud = o3d.io.read_point_cloud(self.ply_file)
        points = np.array(self.cloud.points)
        points = points[points[:,2]<10.00]
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(points)
        new_cloud.paint_uniform_color([0,0,1])
        clean_cloud, ind = new_cloud.remove_radius_outlier(nb_points=20, radius=0.1)
        self.cloud = clean_cloud
        outlier_cloud = new_cloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1,0,0])
        #VISUALIZES AXIS FOR THE POINTCLOUD
        axes = o3d.geometry.LineSet()
        axes.points = o3d.utility.Vector3dVector([[0,0,0],[0,0,100],[0,100,0],[100,0,0]])
        axes.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
        axes.colors = o3d.utility.Vector3dVector([[0,0,1],[0,1,0],[1,0,0]])

        o3d.visualization.draw_geometries([clean_cloud, axes],
                            zoom=0.8,
                            front=[-0.4999, -0.1659, -0.8499],
                            lookat=[2.1813, 2.0619, 2.0999],
                            up=[0.1204, -0.9852, 0.1215])
    
        # If downsampling required:
        # cloud_down = self.cloud.voxel_down_sample(voxel_+size=0.1)

        leftBuilding, rightBuilding = self.ClusterAlongXaxis(clean_cloud)
        leftBuildingpts=np.array(leftBuilding.points)
        rightBuildingpts=np.array(rightBuilding.points)
        
        axes = o3d.geometry.LineSet()
        axes.points = o3d.utility.Vector3dVector([[-1.5,0,0],[-1.5,0,100]])
        axes.lines = o3d.utility.Vector2iVector([[0,1]])
        axes.colors = o3d.utility.Vector3dVector([[1,0,0]])
        o3d.visualization.draw_geometries([leftBuilding, axes],
                            zoom=0.8,
                            front=[-0.4999, -0.1659, -0.8499],
                            lookat=[2.1813, 2.0619, 2.0999],
                            up=[0.1204, -0.9852, 0.1215])
        o3d.visualization.draw_geometries([rightBuilding],
                            zoom=0.8,
                            front=[-0.4999, -0.1659, -0.8499],
                            lookat=[2.1813, 2.0619, 2.0999],
                            up=[0.1204, -0.9852, 0.1215])
        
        points = np.array(clean_cloud.points)
        min = 10000000
        max = -10000000
        for pt in points:
            
            if(max < pt[0]):
                max=pt[0]
            if(min > pt[0]):
                min=pt[0]
        print("max and min", max, min)
        
        
        # OBTAINING THE LEFT AND RIGHT BUILDINGS SEGMENTED IN Z-AXIS
        self.Leftmodels,self.Leftinliers,self.Leftclouds = self.getBuildings(leftBuildingpts,k,l,a)
        # self.Rightmodels,self.Rightinliers,self.Rightclouds = self.getBuildings(rightBuildingpts,k2,l2,a2)

        # Visualization of both segmented buildings
        visualize = Visualization()
        visualize.leftbuildingvisualize(leftBuilding,self.Leftclouds)
        # visualize.rightbuildingvisualize(rightBuilding,self.Rightclouds)
        exit()
        # Iteriate to find roof index
        iRight = self.getRoof(self.Rightinliers,k2)
        iLeft = self.getRoof(self.Leftinliers,k)

        # TODO: Need to modularize these parameters
        m=4;yl=-1.5;yh=1.0;ya=yh-yl/4;offset=0.05

        # Obtaining the points b/w which the distance is to be measured
        listtrueposR = self.selectPoints(self.Rightclouds,m,yl,ya,offset,iRight,"Right")
        listtrueposL = self.selectPoints(self.Leftclouds,m,yl,ya,offset,iLeft,"Left")

        # Visualization and storing the distance b/w points on adjacent buildings
        linepoints = []
        lpts = np.array(self.Leftclouds[iLeft].points)
        rpts = np.array(self.Rightclouds[iRight].points)

        for i in range(4):
            q1 = listtrueposL[i]
            q2 = listtrueposR[i]

            lpt = lpts[q1]
            rpt = rpts[q2]

            lpt = lpt.tolist()
            rpt = rpt.tolist()

            linepoints.append(lpt)
            linepoints.append(rpt)
        
        lines=[[0,1], [2,3], [4,5], [6,7]]
        lineset=o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(linepoints)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.paint_uniform_color([1,0,0])
        
        distlist=[]
        for i in range(4):
            dist = self.find_dist(linepoints[(2*i)],linepoints[(2*i+1)])
            distlist.append(dist*self.scale)
        
        o3d.visualization.draw_geometries([self.Rightclouds[iRight],self.Leftclouds[iLeft], lineset],
                            zoom=0.8,
                            front=[-0.4999, -0.1659, -0.8499],
                            lookat=[2.1813, 2.0619, 2.0999],
                            up=[0.1204, -0.9852, 0.1215])
        return distlist
        
if __name__ == "__main__":
    scale = 7.75
    ply_file = "/home/aditya/IIIT-H/dataset/DJI_0243/dense/0/fused.ply"
    obj = AdjacentRoofDistance(ply_file,scale)
    list=[]
    list = obj.EstimateDistanceAdjacentBuildings()
    print(list[0],list[1],list[2],list[3])