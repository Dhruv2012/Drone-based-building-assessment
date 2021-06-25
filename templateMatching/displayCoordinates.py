import cv2   
import numpy as np   

def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

# driver function
if __name__=="__main__":
  
    # reading the image
    img = cv2.imread(r'F:\IIIT-H Work\win_det_heatmaps\rrcServerData\templateMatching\images\mobilenet\vis_Bakul_003_000470.png', 1)
    
    template = img[345: 496, 385: 495]
    # displaying the image
    # cv2.imshow('template', template)
    cv2.imshow('image', img)

    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
  
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
  
    # close the window
    cv2.destroyAllWindows()

##########RESNET#####################
#Bakul_2_920
# 321.9007633587786, 439.45801526717554
# 695.8473282442748, 857.679389312977

#Bakul_2_970
#[395: 498, 345: 490]

#Bakul_3_470
#538: 621, 703: 845

#Vindhya_002_000320
#205:370, 499: 654


#######SHUFFLENET####################
#Bakul_2_970
#[2:126, 730: 880]

#######mobilenet#####################

#Vindhya02_320
#[210:365, 499: 645]

#Bakul_3_470
[106:249, 346:500]