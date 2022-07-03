from distutils.command.build_scripts import first_line_re
import cv2
from matplotlib import widgets
from matplotlib.pyplot import get
import numpy as np
import pandas as pd
import geopy.distance
import open3d as o3d

import utils

def stitch(img1, img2):
    # img1 = cv2.resize(img1,(0,0),fx = 0.2, fy = 0.2)
    # img2 = cv2.resize(img2,(0,0),fx = 0.2, fy = 0.2)
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.AKAZE_create()
    kps1, features1 = descriptor.detectAndCompute(img_gray1, None)
    kps2, features2 = descriptor.detectAndCompute(img_gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    best_match = bf.knnMatch(features1, features2, 2)
    good_match = []
    for m, n in best_match:
        if m.distance < n.distance * 0.7:
            good_match.append(m)

    img3 = cv2.drawMatches(img1, kps1, img2, kps2, good_match, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("img3",img3)
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])
    h, w, c = img1.shape
    # warp = cv2.warpPerspective(img1, H, (2*w, 2*h))
    # warp[:img2.shape[0], :img2.shape[1]] = img2
    H =np.eye(3)

    if len(good_match) > 4:
        pts1 = np.float32([kps1[m.queryIdx] for m in good_match])
        pts2 = np.float32([kps2[m.trainIdx] for m in good_match])
        H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4)
    warp = cv2.warpPerspective(img1, H, (2*w, 2*h))
    warp[:img2.shape[0], :img2.shape[1]] = img2
    return warp, H

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

    print("t1\n", o_t_a, "\nt2\n", o_t_b)
    print("Rotation\n",b_R_a, "\ntranslation\n", b_t_a)
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
    K[0,0] = K[1,1] = 1311*0.2
    K[0,2] = 960*0.2
    K[1,2] = 540*0.2
    pt_list = []
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0,0,0],[0,0,100],[0,100,0],[100,0,0]])
    axes.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3]])
    axes.colors = o3d.utility.Vector3dVector([[0,0,1],[0,1,0],[1,0,0]])
    warp_img = np.zeros((10,10))
    warpfm = np.zeros((10,10))
    stitcher = cv2.Stitcher.create(mode =1)
    opt_img = np.zeros((3,3))
    good_img = np.zeros((3,3))
    img_list = []
    my_list = []
    for i in transformation_matrices.keys():
        if(count==0):
            o_tf_prev = transformation_matrices[i][0]
            img_path_prev = transformation_matrices[i][1]
            prev_d = transformation_matrices[i][2]
            prev_frame = cv2.imread('/home/kuromadoshi/IIITH/'+img_path_prev.split("/")[4] + '/' + img_path_prev.split("/")[5] + '/' + img_path_prev.split("/")[6] + '/' + img_path_prev.split("/")[7])
            prev_frame = cv2.resize(prev_frame,(0,0),fx = 0.2,fy = 0.2)
            
            # opt_img = np.zeros(shape = (int(5*prev_frame.shape[0]), int(5*prev_frame.shape[1]), prev_frame.shape[2]))
            H = np.eye(3)
            warp_img = cv2.warpPerspective(prev_frame, H, (2*prev_frame.shape[1], 2*prev_frame.shape[0]))
            warpfm = cv2.warpPerspective(prev_frame, H, (2*prev_frame.shape[1], 2*prev_frame.shape[0]))
            opt_img = prev_frame
            good_img = opt_img
            img_list.append(good_img)
            my_list.append(good_img)
            count+=1
            continue
        o_tf_cur = transformation_matrices[i][0]
        img_path_cur = transformation_matrices[i][1]
        cur_d = transformation_matrices[i][2]
        cur_frame = cv2.imread('/home/kuromadoshi/IIITH/'+img_path_cur.split("/")[4] + '/' + img_path_cur.split("/")[5] + '/' + img_path_cur.split("/")[6] + '/' + img_path_cur.split("/")[7])
        cur_frame = cv2.resize(cur_frame,(0,0),fx = 0.2,fy = 0.2)
        cur_n = np.array([0,0, -1])
        cur_n = cur_n.reshape(3,1)
        # cur_H_prev_eu = getHomography(o_tf_prev, o_tf_cur, cur_n, prev_d)
        # cur_H_prev = (K.dot(cur_H_prev_eu)).dot(np.linalg.inv(K))
        # cur_H_prev /= cur_H_prev[2,2]
        # Hi = np.linalg.inv(cur_H_prev)
        # Hi /= Hi[2,2]
        # print()
        # print("curTf\n", o_tf_cur)
        # print("Fm H")
    
        # _, Hfm = stitch(prev_frame, cur_frame)
        # print(Hfm)
        # print("Hi")
        # print(Hi)
        # print("Homography =")
        # print(cur_H_prev)
        # print("Homo eu")
        # print(cur_H_prev_eu)
       
        
        print("K")
        print(K)
        img_list.append(cur_frame)
        my_list.append(cur_frame)
        # status, opt_img = stitcher.stitch(img_list)
        # warpfm = cv2.warpPerspective(warpfm, Hfm , (prev_frame.shape[1] + cur_frame.shape[1], 2000), flags = cv2.INTER_NEAREST)
        # warp_img = cv2.warpPerspective(warp_img, cur_H_prev, (prev_frame.shape[1] + cur_frame.shape[1], 2000), flags = cv2.INTER_NEAREST)
        # warp_img[:cur_frame.shape[0], :cur_frame.shape[1]] = cur_frame
        # warpfm[:cur_frame.shape[0], :cur_frame.shape[1]] = cur_frame
        # # w2 = cv2.warpPerspective(cur_frame, np.eye(3), (opt_img.shape[1], opt_img.shape[0]), flags = cv2.INTER_LINEAR)
        # opt_img[warped_img > 0] = warped_img[warped_img > 0]
        # cv2.imshow("warpimg", warp_img)
        # cv2.imshow("Warpfm", warpfm)
        # print("status=",status)
        # if(status ==0):
        #     # cv2.imshow("stitch", opt_img)
        #     # cv2.waitKey(0)
        #     img_list = [good_img, cur_frame]
        #     good_img = opt_img
        #     img_list = [good_img]
        # else:
        #     img_list.append(prev_frame)
        # cv2.destroyAllWindows()
        
        pt = [o_tf_cur[0,3], o_tf_cur[1,3], o_tf_cur[2,3]]
        pt_list.append(pt)

        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(pt_list)
        pcl.paint_uniform_color([1,0,0])
        colors = np.array(pcl.colors)
        colors[colors.shape[0]-1] = np.array([0,0,1])
        pcl.colors = o3d.utility.Vector3dVector(colors)
        # o3d.visualization.draw_geometries([pcl, axes],
        #                     zoom=0.8,
        #                     front=[-0.4999, -0.1659, -0.8499],
        #                     lookat=[2.1813, 2.0619, 2.0999],
        #                     up=[0.1204, -0.9852, 0.1215])

        o_tf_prev = o_tf_cur
        img_path_prev = img_path_cur
        prev_d = cur_d
        prev_frame = cur_frame
        print("Hello")
        print("count = ", count)
        count+=1
        if(count ==170):
            break

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(pt_list)
    pcl.paint_uniform_color([1,0,0])
    stitcher = cv2.Stitcher.create(mode = 0)
    status, opt = stitcher.stitch(my_list)
    cv2.imwrite("myopt.png", opt)
    # cv2.imwrite("svo.png", warp_img)
    # cv2.imwrite("fm.png", warpfm)
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








    







    
