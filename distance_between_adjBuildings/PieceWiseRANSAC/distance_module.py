from tkinter import Tk, filedialog
import cv2
import numpy as np

class Distance_Module(object):

    scale = 7.75 

    img_txt_path = "/home/kuromadoshi/IIITH/workspace/ws1/sparse/0/images.txt"
    img_folder_path = "/home/kuromadoshi/IIITH/workspace/dense/0/"
    depth_folder_path = ""
    camera_txt_path = ""
    drone_K_mat = 4
    R = 1
    t = 1
    def __init__(self):

        root = Tk()
        root.withdraw()

        self.img_txt_path = "/home/kuromadoshi/IIITH/workspace/ws1/sparse/0/images.txt"
        self.camera_txt_path = "/home/kuromadoshi/IIITH/workspace/ws1/sparse/0/cameras.txt"
        self.img_folder_path = "/home/kuromadoshi/IIITH/workspace/ws1/dense/0/images/"
        self.depth_folder_path = "/home/kuromadoshi/IIITH/workspace/ws1/dense/0/stereo/depth_maps/"
        


        # self.img_text_path = filedialog.askdirectory(title = "chose the image text directory")
        # self.img_folder_path = filedialog.askdirectory(title = "choose the image directory")
        # self.img_folder_path = filedialog.askdirectory(title = "choose the depth image directory")
    
    def getR_from_q(self, q):
        # w, x, y, z
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        return np.array([
            [-1 + 2 * q[0] ** 2 + 2 * q[1] ** 2,
            2 * q[1] * q[2] - 2 * q[0] * q[3],
            2 * q[3] * q[1] + 2 * q[0] * q[2]],

            [2 * q[1] * q[2] + 2 * q[0] * q[3],
            -1 + 2 * q[0] ** 2 + 2 * q[2] ** 2,
            2 * q[2] * q[3] - 2 * q[0] * q[1]],

            [2 * q[3] * q[1] - 2 * q[0] * q[2],
            2 * q[2] * q[3] + 2 * q[0] * q[1],
            -1 + 2 * q[0] ** 2 + 2 * q[3] ** 2]])
    def getH_from_R_t(self, R, t):
	    # assuming R, t are numpy array
        h = np.column_stack((R, t))
        a = np.array([0, 0, 0, 1])
        h = np.vstack((h, a))
        assert h.shape == (4,4)
        return h
    def getH_Inverse_from_R_t(self, R, t):
        # assuming R, t are numpy array
        h = np.column_stack((R.T, -R.T@t))
        a = np.array([0, 0, 0, 1])
        h = np.vstack((h, a))
        assert h.shape == (4,4)
        return h
    def SelectPointsInImage(self, PathIn, Image=None):
        # Returns the selected points in an image in list
        positions, click_list = [], []

        def callback(event, x, y, flags, param):
            if event == 1:
                
                click_list.append((x,y))
        cv2.namedWindow('SelectPoints')
        cv2.setMouseCallback('SelectPoints', callback)
        if PathIn is not None:
            Image = cv2.imread(PathIn)	# Don't resize, otherwise scale the points accordingly.
        while True:
            cv2.imshow('SelectPoints', Image)
            k = cv2.waitKey(1)
            if k == 27:
                break
        cv2.destroyAllWindows()
        return click_list
    
    def DisplaySelectedContour(self, image_path, click_list):
        image = cv2.imread(image_path)
    
        for i in range(len(click_list)-1):
            cv2.line(image, click_list[i],click_list[i+1],(0,0,255),2)

        while True:
            cv2.imshow('Selected Countour (PRESS Esc TO EXIT)',image)
            k = cv2.waitKey(1)

            if k == 27:
                break
        cv2.destroyAllWindows()

    def ReadDepthMap(self, path):
        min_depth_percentile = 5
        max_depth_percentile = 95

        with open(path, "rb") as depthfile:
            width, height, channels = np.genfromtxt(depthfile, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
            depthfile.seek(0)
            num_delimiter = 0
            byte = depthfile.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = depthfile.read(1)
            array = np.fromfile(depthfile, np.float32)
        array = array.reshape((width, height, channels), order="F")

        depth_map = np.transpose(array, (1, 0, 2)).squeeze()

        min_depth, max_depth = np.percentile(depth_map, [min_depth_percentile, max_depth_percentile])
        print(min_depth, max_depth)

        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth

        return depth_map
    
    def ReadCameraOrientation(self,pathIn, findAll=True, findID=None, findName=None):
        """
            1. Returns the R, t to transform from world frame to camera frame.
            2. If findAll==false, returns the findID camera R,t
            Not optimized for task 2 alone. 
        """
        ID_Rt = {} # if only few cam R,t required.
        Name_Rt = {}
        with open(pathIn) as f:
            lines = f.readlines()
        # print(len(lines))

        line_count = 0 # Every odd line needs to be skipped, it has 2D points(not using right now).
        Rs = []
        ts = []
        only_transformation_lines = []

        for index, line in enumerate(lines):
            line = line.strip()

            if not line.startswith('#'):
                line_count = line_count + 1

                if line_count % 2 == 1:
                    elements = line.split(" ")
                    only_transformation_lines.append(elements)

        # print(only_transformation_lines)
        only_transformation_lines.sort(key=lambda x: int(x[0]))

        old_H = np.eye(4) # Identity transformation

        # This should not be running everytime.
        for line in only_transformation_lines:
            ID = int(line[0])
            Name = line[9]
            q = []
            for i in range(1,5):
                q.append(float(line[i]))
            t = []
            for j in range(5,8):
                t.append(float(line[j]))
            # print(q, t)

            R = self.getR_from_q(q)
            R.shape = (3,3)
            t = (np.array(t)).T
            t.shape = (3,1)
            H = self.getH_from_R_t(R, t)
            old_H = H@old_H
            # print(R)
            # print(t)
            Rs.append(R)
            ts.append(t)
            ID_Rt[ID] = [R, t, old_H]
            Name_Rt[Name] = [R, t, old_H]
        
        if findAll:
            return Rs, ts
        else:
            if findID is not None:
                return ID_Rt[findID][0], ID_Rt[findID][1], ID_Rt[findID][2], ID_Rt
            else:
                return Name_Rt[findName][0], Name_Rt[findName][1], Name_Rt[findName][2], Name_Rt

    def read_camera_intrinsics(self, camera_txt_path, image_num):
        with open(camera_txt_path) as f:
            lines = f.readlines()
        line_count=0
        intrinsic_mat = np.identity(3)
        for index, line in enumerate(lines):
            line = line.strip()
            if not line.startswith(str(image_num)):
                line_count+=1
            else:
                elements = line.split(" ")
                intrinsic_mat[0,0]=intrinsic_mat[1,1]=elements[4] # focal length
                intrinsic_mat[0,2]=elements[5] # principal point cx
                intrinsic_mat[1,2]=elements[6] # principal point cy
        return intrinsic_mat
 

    



    def Get3Dfrom2D(self, List2D, K, R, t, d, H=None):
        # List2D : n x 2 array of pixel locations in an image
        # K : Intrinsic matrix for camera
        # R : Rotation matrix describing rotation of camera frame
        # 	  w.r.t world frame.
        # t : translation vector describing the translation of camera frame
        # 	  w.r.t world frame
        # [R t] combined is known as the Camera Pose.

        List2D = np.array(List2D)
        List3D = []
        d = np.array(d)
        inv_trans = -1*np.dot(R.T,t)
        # t.shape = (3,1)

        # if H is not None:
        # 	R = H[:3,:3]
        # 	t = H[:3,3]
        print(List2D.shape)
        for p, q  in zip(List2D, d):
            # Homogeneous pixel coordinate
            p = np.array([p[0], p[1], 1]).T; p.shape = (3,1)
            # print("pixel: \n", p)

            # Transform pixel in Camera coordinate frame
            pc = np.linalg.inv(K) @ p
            # print("pc : \n", pc, pc.shape)

            # Transform pixel in World coordinate frame
            pw = np.dot(R.T, pc) + inv_trans
            # print("pw : \n", pw, t.shape, R.shape, pc.shape)

            # Transform camera origin in World coordinate frame
            cam = np.array([0,0,0]).T; cam.shape = (3,1)
            cam_world = np.dot(R.T, cam) + inv_trans
            # print("cam_world : \n", cam_world)

            # Find a ray from camera to 3d point
            vector = pw - cam_world
            unit_vector = vector / np.linalg.norm(vector)
            # print("unit_vector : \n", unit_vector)
            
            # Point scaled along this ray
            p3D = cam_world + q * unit_vector
            # print("p3D : \n", p3D)
            List3D.append(p3D)

        return List3D

    def DepthValues(self, List2D, depth_map):
        depth_values = []

        for p in List2D:
            depth_values.append(depth_map[p[1], p[0]])

        return depth_values

    def GetMeshDistance(self, points3d1, points3d2): # TODO: IS this correct??

        mean1 = np.mean(points3d1, axis=0)
        mean2 = np.mean(points3d2, axis=0)

        mesh_distance = ((mean1-mean2)**2).sum()**0.5
        return mesh_distance
    def ReadCameraOrientation(self, pathIn, findAll=True, findID=None, findName=None):
        """
            1. Returns the R, t to transform from world frame to camera frame.
            2. If findAll==false, returns the findID camera R,t
            Not optimized for task 2 alone. 
        """
        ID_Rt = {} # if only few cam R,t required.
        Name_Rt = {}
        with open(pathIn) as f:
            lines = f.readlines()
        # print(len(lines))

        line_count = 0 # Every odd line needs to be skipped, it has 2D points(not using right now).
        Rs = []
        ts = []
        only_transformation_lines = []

        for index, line in enumerate(lines):
            line = line.strip()

            if not line.startswith('#'):
                line_count = line_count + 1

                if line_count % 2 == 1:
                    elements = line.split(" ")
                    only_transformation_lines.append(elements)

        # print(only_transformation_lines)
        only_transformation_lines.sort(key=lambda x: int(x[0]))

        old_H = np.eye(4) # Identity transformation

        # This should not be running everytime.
        for line in only_transformation_lines:
            ID = int(line[0])
            Name = line[9]
            q = []
            for i in range(1,5):
                q.append(float(line[i]))
            t = []
            for j in range(5,8):
                t.append(float(line[j]))
            # print(q, t)

            R = self.getR_from_q(q)
            R.shape = (3,3)
            t = (np.array(t)).T
            t.shape = (3,1)
            H = self.getH_from_R_t(R, t)
            old_H = H@old_H
            # print(R)
            # print(t)
            Rs.append(R)
            ts.append(t)
            ID_Rt[ID] = [R, t, old_H]
            Name_Rt[Name] = [R, t, old_H]
        
        if findAll:
            return Rs, ts
        else:
            if findID is not None:
                return ID_Rt[findID][0], ID_Rt[findID][1], ID_Rt[findID][2], ID_Rt
            else:
                return Name_Rt[findName][0], Name_Rt[findName][1], Name_Rt[findName][2], Name_Rt
            

    def perform3dFrom2d(self,img_name,depthimg_bin_name):
        
        depth_map_path = self.depth_folder_path+depthimg_bin_name
        print(type(self.depth_folder_path))
        print(type(depth_map_path))
        image_path = self.img_folder_path + img_name
        depth_map=self.ReadDepthMap(depth_map_path)    #helper

        print(image_path)
        print(depth_map_path)
        
        R, t, _, _ = self.ReadCameraOrientation(self.img_txt_path, False, None, img_name) #helper
        print(R,"\n",t)
        print(img_name.split('.')[0])
        # exit()
        drone_k = self.read_camera_intrinsics(self.camera_txt_path,img_name.split('.')[0])
        print(drone_k,"\n")
    
        List2D =self. SelectPointsInImage(image_path) # Helper
        self.DisplaySelectedContour(image_path, List2D) #]
        
        
        
        depth_values = self.DepthValues(List2D, depth_map) #
        print(len(depth_values))
        
        List3D = self.Get3Dfrom2D(List2D, drone_k, R, t, depth_values) #helper
        print(len(List3D))
        List3D = np.concatenate(List3D, axis=1).T
        return List3D

    def run(self):


        image1_name="10.jpg"
        image2_name="23.jpg"
        depthimg1_bin_name="10.jpg.geometric.bin"
        depthimg2_bin_name="23.jpg.geometric.bin"
        List3D_1= self.perform3dFrom2d(image1_name, depthimg1_bin_name)
        List3D_2=self.perform3dFrom2d(image2_name, depthimg2_bin_name)

        mesh_distance = self.GetMeshDistance(List3D_1, List3D_2)
        actual_distance =mesh_distance*self.scale
        print("Mesh Distance:",mesh_distance)
        print("Distance:", actual_distance)
apple = Distance_Module()
apple.run()