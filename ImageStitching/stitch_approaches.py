from venv import create
import cv2
from matplotlib import widgets
from matplotlib.pyplot import get
import numpy as np
import pandas as pd

def extract_features(img):
    
    #Using Clahe for better contrast, thus increasing the number of features detected
#     clahe = cv2.createCLAHE(clipLimit=25.0)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img=clahe.apply(img)
    
    #Using FAST
    fast= cv2.FastFeatureDetector_create(threshold = 25, nonmaxSuppression = True)
    kp = fast.detect(img)
    kp = np.array([kp[idx].pt for idx in range(len(kp))], dtype = np.float32)




def track_features(image_ref, image_cur,ref):
    #Initializing LK parameters
    lk_params = dict(winSize=(21, 21), criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref, None, **lk_params)
    
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image_cur, image_ref, kp2, None, **lk_params)
#     distance=abs(ref-kp1).max(-1)

    return kp1, kp2
def stitch(img1, img2):
    # img1 = cv2.resize(img1,(0,0),fx = 0.2, fy = 0.2)
    # img2 = cv2.resize(img2,(0,0),fx = 0.2, fy = 0.2)
    # img_gray1 = img1
    # img_gray2 = img2
    img_gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img_gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.ORB_create()
    kps1, features1 = descriptor.detectAndCompute(img_gray1, None)
    kps2, features2 = descriptor.detectAndCompute(img_gray2, None)

    kps1_ref = np.array([kps1[idx].pt for idx in range(len(kps1))], dtype = np.float32)
    kps1_of, kps2_of = track_features(img1, img2, kps1_ref)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)

    best_match = bf.knnMatch(features1, features2, 2)
    good_match = []
    for m, n in best_match:
        if m.distance < n.distance * 0.7:
            good_match.append(m)
    # print(len(good_match))
    img3 = cv2.drawMatches(img1, kps1, img2, kps2, good_match, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("img3",img3)
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])
    # h, w, = img1.shape
    # warp = cv2.warpPerspective(img1, H, (2*w, 2*h))
    # warp[:img2.shape[0], :img2.shape[1]] = img2
    H =np.eye(3)
    A = np.eye(3)
    if len(good_match) > 4:
        pts1 = np.float32([kps1[m.queryIdx] for m in good_match])
        pts2 = np.float32([kps2[m.trainIdx] for m in good_match])
        A, inliner = cv2.estimateAffine2D(pts1, pts2)
    else:
        print("oops homo esti failed")
    A_,inl = cv2.estimateAffine2D(kps1_of, kps2_of)
    # A_, inl = cv2.findHomography(kps1_of, kps2_of)
    print("A_", A_)
    print("A = ", A)
    return A_

if __name__ == "__main__":
    # print("hello")
    count = 0
    H = np.eye(3)
    H_old = H
    num_img = 40
    # k = 100
    for i in range(0, num_img, 1):
        
        j = i+1
        # if(j>=979):
        #     continue
        if(count == 0):
            prev_frame = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/Input/' + str(j) + '.jpg')
            prev_mask = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/masks/' + str(j) + '.jpg')
            prev_roof_mask = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/roof_masks/' + str(j) + '.png')
        
            print('/home/kuromadoshi/IIITH/dataset/DJI_0641/images/' + str(j) + '.jpg')
            prev_frame = cv2.resize(prev_frame, (0,0), fx = 0.2, fy = 0.2)
            prev_mask = cv2.resize(prev_mask, (0, 0), fx = 0.2, fy = 0.2)
            prev_roof_mask = cv2.resize(prev_roof_mask, (384, 216))

            count += 1
            warp_naive = prev_frame #cv2.warpAffine(prev_frame, np.eye(3), (prev_frame.shape[1] + prev_frame.shape[1], 2000), flags = cv2.INTER_NEAREST)
            warp_mask = prev_mask
            warp_roof_mask = prev_roof_mask
            continue
        cur_frame = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/Input/' + str(j) + '.jpg')
        cur_mask = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/masks/' + str(j) + '.jpg')
        cur_roof_mask = cv2.imread('/home/kuromadoshi/IIITH/dataset/DJI_0641_mask/roof_masks/' + str(j) + '.png')
        print('/home/kuromadoshi/IIITH/dataset/endgame/images/' + str(j) + '.jpg')
        cur_frame = cv2.resize(cur_frame, (0,0), fx = 0.2, fy = 0.2)
        cur_mask = cv2.resize(cur_mask, (0, 0), fx = 0.2, fy = 0.2)
        cur_roof_mask = cv2.resize(cur_roof_mask, (384, 216))
        print("hello",cur_frame.shape)
        print("not hello", cur_roof_mask.shape)
        A = stitch(prev_frame, cur_frame)
        # H[:2, :3] = A
        # H_cur = H_old.dot(H)
        # print(H_cur, A)
        warp_naive = cv2.warpAffine(warp_naive, A, (prev_frame.shape[1] + cur_frame.shape[1], 1000), flags = cv2.INTER_NEAREST)
        warp_mask = cv2.warpAffine(warp_mask, A, (prev_mask.shape[1] + prev_frame.shape[1], 1000), flags = cv2.INTER_NEAREST)
        warp_roof_mask = cv2.warpAffine(warp_roof_mask, A, (prev_mask.shape[1] + prev_frame.shape[1], 1000), flags = cv2.INTER_NEAREST)
        # warp_naive = cv2.warpPerspective(warp_naive, A, (prev_frame.shape[1] + cur_frame.shape[1], 3000), flags = cv2.INTER_NEAREST)
        warp_naive[:cur_frame.shape[0], :cur_frame.shape[1]] = cur_frame
        # warp_mask[:cur_mask.shape[0], :cur_mask.shape[1]] = cur_mask
        # warp_naive[:cur_frame.shape[0], :cur_frame.shape[1]] = np.maximum(warp_naive[:cur_frame.shape[0], :cur_frame.shape[1]],  cur_frame)
        warp_mask[:cur_mask.shape[0], :cur_mask.shape[1]] = np.maximum(warp_mask[:cur_mask.shape[0], :cur_mask.shape[1]],  cur_mask)
        warp_roof_mask[:cur_roof_mask.shape[0], :cur_roof_mask.shape[1]] = np.maximum(warp_roof_mask[:cur_roof_mask.shape[0], :cur_roof_mask.shape[1]],  cur_roof_mask)
        # warp_2 = cv2.warpPerspective(warp_2, H_cur, (prev_frame.shape[1] + cur_frame.shape[1], 3000), flags = cv2.INTER_NEAREST)
        # warp_2[:cur_frame.shape[0], :cur_frame.shape[1]] = cur_frame
        prev_frame = cur_frame
        prev_mask = cur_mask
        prev_roof_mask = cur_roof_mask
        cv2.imshow("warp", warp_mask)
        cv2.imshow("wapper", warp_naive)
        cv2.imshow("roof", warp_roof_mask)
        # cv2.imshow("warp2", warp_2)
        cv2.waitKey(100)

        
    cv2.imwrite('orange_max.png', warp_naive)
    cv2.imwrite('orange_roof_mask.png', warp_roof_mask)
    cv2.imwrite('orange_kamask.png', warp_mask)

