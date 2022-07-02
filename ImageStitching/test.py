from json import detect_encoding
import cv2
import numpy as np
img = np.ones((100,100,3))
H = np.eye(3)
H[0,2] = 50
H[1,2] = 100
warp = cv2.warpPerspective(img, H, (200,200))
print(img.shape)
print(warp.shape)

img1 = cv2.imread("/home/kuromadoshi/IIITH/dataset/DJI_0641/images/1.jpg")
img2 = cv2.imread("/home/kuromadoshi/IIITH/dataset/DJI_0641/images/51.jpg")
img1 = cv2.resize(img1,(0,0),fx = 0.2, fy = 0.2)
img2 = cv2.resize(img2,(0,0),fx = 0.2, fy = 0.2)
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

kps1 = np.float32([kp.pt for kp in kps1])
kps2 = np.float32([kp.pt for kp in kps2])

if len(good_match) > 4:
    pts1 = np.float32([kps1[m.queryIdx] for m in good_match])
    pts2 = np.float32([kps2[m.trainIdx] for m in good_match])
    H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4)
print(H)
cv2.waitKey(0)

h, w, c = img1.shape
warp = cv2.warpPerspective(img1, H, (2*w, 2*h))

warp[:img2.shape[0], :img2.shape[1]] = img2

K = np.eye(3)
K[0,0] = K[1,1] = 1308
K[0,2] = 960
K[1,2] = 540

H_dash = (np.linalg.inv(K).dot(H)).dot(K)
print("K\n", K)
print("Hd\n",H_dash)
# cv2.imshow("img", img)
cv2.imshow("img1",img_gray1)
cv2.waitKey(0)
cv2.imshow("img2", img_gray2)
cv2.imshow("match",img3)
cv2.imshow("warp", warp)
cv2.waitKey(0)
