import os
import cv2
import numpy as np

rgbImgDir = r"F:\IIIT-H Work\win_det_heatmaps\datasets\IIIT-H Dataset\GoogleEarth\instance_segmentation\dataset_voc\SegmentationClassPNG"
binaryImgDir = r"F:\IIIT-H Work\win_det_heatmaps\datasets\IIIT-H Dataset\GoogleEarth\instance_segmentation\dataset\labels"

allfiles = os.listdir(rgbImgDir)
images = [ fname for fname in allfiles if fname.endswith('.png')]

for i in range(len(images)):
    path = os.path.join(rgbImgDir, images[i])
    print('path:', path)
    inputImg = cv2.imread(path)
    # print('inputImg shape:', inputImg.shape)
    # print('inputImg datatype:', inputImg.dtype)

    # print('inputImg', np.amax(inputImg, axis=2))
    grayImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2GRAY)
    # print(np.amax(grayImg))
    _, binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY)

    print(np.amax(binaryImg))
    cv2.imwrite(os.path.join(binaryImgDir, images[i]), binaryImg)