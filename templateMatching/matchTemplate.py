import cv2   
import numpy as np   


fileName = 'vis_Bakul_003_000470.png'
imgDirPath = 'F:\\IIIT-H Work\\win_det_heatmaps\\rrcServerData\\templateMatching\\images\\mobilenet\\'
imgPath = imgDirPath + fileName
# Reading the main image   
img_rgb = cv2.imread(imgPath,1)  
# It is need to be convert it to grayscale   

# Read the template   
template = img_rgb[106:249, 346:500]
print(template.shape)
cv2.imshow('template', template)  
searchImg = img_rgb[106:249, : ,:]


img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)   
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)   
searchImg_gray = cv2.cvtColor(searchImg, cv2.COLOR_BGR2GRAY)

print('img_grayshape', img_gray.shape)

print(template_gray.shape)
# Store width in variable w and height in variable h of template  
w, h = template_gray.shape[::-1]   
# Now we perform match operations.   
res = cv2.matchTemplate(searchImg_gray,template_gray,cv2.TM_CCOEFF_NORMED)   
# Declare a threshold   
threshold = 0.6
# Store the coordinates of matched region in a numpy array   
loc = np.where( res >= threshold)   
# Draw a rectangle around the matched region.   
for pt in zip(*loc[::-1]):   
    cv2.rectangle(searchImg, pt, (pt[0] + w, pt[1] + h), (255,0,0), 1)   
# Now display the final matched template image   
cv2.imshow('Detected',searchImg)  
cv2.imwrite('Template_' +  '0.6_threshold_' + fileName, template)
cv2.imwrite('TemplateMatched_'  + '0.6_threshold_' + fileName, searchImg)
# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()