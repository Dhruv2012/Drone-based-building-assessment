from distutils.command.build_scripts import first_line_re
import cv2
from matplotlib import widgets
from matplotlib.pyplot import get
import numpy as np
import pandas as pd
import geopy.distance
import open3d as o3d

import utils


def getHomography(o_tf_a,o_tf_b, a_n, a_d):
    # ra_b is the transformation to a frame from b frame.
    o_R_a = o_tf_a[0:3, 0:3]
    o_R_b = o_tf_b[0:3, 0:3]
    o_t_a = o_tf_a[0:3, 3:4]
    o_t_b = o_tf_b[0:3, 3:4]

    a_R_o = o_R_a.T
    b_R_o = o_R_b.T
    a_t_o = -o_t_a
    b_t_o = -o_t_b

    b_R_a = b_R_o.dot(a_R_o.T)      
    b_t_a = b_R_o.dot(-a_R_o.T.dot(a_t_o)) + b_t_o
    print(b_R_a, b_t_a)
    print((b_t_a.dot(a_n.T))/a_d)
    b_H_a = b_R_a -(b_t_a.dot(a_n.T))/a_d
    return b_H_a

if __name__ == "__main__":
    
    # o_tf_a = np.eye(4)
    # o_tf_b = np.eye(4)

    # o_tf_a[1,3] = -1
    # o_tf_a[2,3] = -2

    # o_tf_b[1,3] = -5
    # o_tf_b[2,3] = -2

    # a_n = np.array([0,0, -1])
    # a_n = a_n.reshape(3,1)
    # a_d = 2
    # H = getHomography(o_tf_a, o_tf_b, a_n, a_d)
    # print(H)
    # print(a_n,a_n.shape)
    # exit()
    orb_data = utils.ORBReader('/home/kuromadoshi/IIITH/dataset/DJI_0641/ORB_DJI0641.txt')
    log_data = pd.read_csv('/home/kuromadoshi/IIITH/dataset/DJI_0641/data.csv')
    
    combined_logs = utils.Combinelogs(log_data, orb_data)
    transformation_matrices = utils.GetHomographyData(combined_logs)


    count = 0
  
    o_tf_prev = np.zeros((4,4))
    img_path_prev = ""
    prev_d = 0
    prev_frame = np.zeros((3,3))
    
    opt_img = np.zeros((10,10))
    K = np.eye(3)
    K[0,0] = K[1,1] = 1308*0.2
    K[0,2] = 960*0.2
    K[1,2] = 540*0.2
    pt_list = []
    for i in transformation_matrices.keys():
        if(count==0):
            o_tf_prev = transformation_matrices[i][0]
            img_path_prev = transformation_matrices[i][1]
            prev_d = transformation_matrices[i][2]
            prev_frame = cv2.imread('/home/kuromadoshi/IIITH/'+img_path_prev.split("/")[4] + '/' + img_path_prev.split("/")[5] + '/' + img_path_prev.split("/")[6] + '/' + img_path_prev.split("/")[7])
            prev_frame = cv2.resize(prev_frame,(0,0),fx = 0.2,fy = 0.2)
            
            opt_img = np.zeros(shape = (int(5*prev_frame.shape[0]), int(5*prev_frame.shape[1]), prev_frame.shape[2]))
            count+=1
            continue
        o_tf_cur = transformation_matrices[i][0]
        img_path_cur = transformation_matrices[i][1]
        cur_d = transformation_matrices[i][2]
        cur_frame = cv2.imread('/home/kuromadoshi/IIITH/'+img_path_cur.split("/")[4] + '/' + img_path_cur.split("/")[5] + '/' + img_path_cur.split("/")[6] + '/' + img_path_cur.split("/")[7])
        cur_frame = cv2.resize(cur_frame,(0,0),fx = 0.2,fy = 0.2)
        cur_n = np.array([0,0, -1])
        cur_n = cur_n.reshape(3,1)
        prev_H_cur_eu = getHomography(o_tf_cur, o_tf_prev, cur_n, cur_d)
        prev_H_cur = (K.dot(prev_H_cur_eu)).dot(np.linalg.inv(K))
        prev_H_cur /= prev_H_cur[2,2]

        print("Homography =")
        print(prev_H_cur)
        print("Homo eu")
        print(prev_H_cur_eu)
        Hi = np.linalg.inv(prev_H_cur)
        print(Hi)
        print("K")
        print(K)
        warped_img = cv2.warpPerspective(prev_frame, Hi, (prev_frame.shape[1]+cur_frame.shape[1], prev_frame.shape[0]+cur_frame.shape[0]), flags = cv2.INTER_LINEAR)
        warped_img[:prev_frame.shape[0], :prev_frame.shape[1]] = cur_frame
        # # w2 = cv2.warpPerspective(cur_frame, np.eye(3), (opt_img.shape[1], opt_img.shape[0]), flags = cv2.INTER_LINEAR)
        # opt_img[warped_img > 0] = warped_img[warped_img > 0]
        cv2.imshow(img_path_cur, warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pt = [o_tf_cur[0,3], o_tf_cur[1,3], o_tf_cur[2,3]]
        pt_list.append(pt)
        o_tf_prev = o_tf_cur
        img_path_prev = img_path_cur
        prev_d = cur_d
        prev_frame = cur_frame
        print("Hello")
        print("count = ", count)
        if(count ==1):
            break

    # pcl = o3d.geometry.PointCloud()
    # pcl.points = o3d.utility.Vector3dVector(pt_list)
    # pcl.paint_uniform_color([1,0,0])
    # axes = o3d.geometry.LineSet()
    # axes.points = o3d.utility.Vector3dVector([[0,0,0],[0,0,100],[0,100,0],[100,0,0]])
    # axes.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
    # axes.colors = o3d.utility.Vector3dVector([[0,0,1],[0,1,0],[1,0,0]])

    # o3d.visualization.draw_geometries([pcl, axes],
    #                         zoom=0.8,
    #                         front=[-0.4999, -0.1659, -0.8499],
    #                         lookat=[2.1813, 2.0619, 2.0999],
    #                         up=[0.1204, -0.9852, 0.1215])


    # result=cv2.warpPerspective(img[0],h,(width,height))
    # print(result.shape)
    # # #Till now we have wrapped the img1 in the same plane as img2 and now we will place img2 in the resulting img
    # cv2.imshow("only transformed", result)
    # z = np.zeros((216, 384, 3))
    # result[0:img[1].shape[0],0:img[1].shape[1]]=z

    # cv2.imshow('Combined Image',result)
    # while(cv2.waitKey(1)!=ord('q')):
    #     pass
    # cv2.destroyAllWindows()








    







    
