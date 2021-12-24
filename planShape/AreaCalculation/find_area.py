import numpy as np 
import cv2
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon


focalLength = 2.96347438e+03
depth = 100

'''
Height Info:
17:20:36: 5th image -> 100 meters
17:20:32: 4th image -> 93 meters
17:20:30: 3rd image -> 90.4 meters
17:20:21: 2nd image -> 70.2 meters
17:20:19: 1st image -> 69 meters

'''
def get_z(pt1, pt2, h, f1, f2):
    '''
    pt1: y coord of Point1 in Image1
    pt2: y coord of Point1 in Image2
    '''
    # print('pt1:', pt1)
    # print('pt2:', pt2)
    Z = (((f1*pt1*h)/(f2*pt1-f1*pt2)))
    return Z

def distance(pt1, pt2):
    x1,y1 = pt1
    x2,y2 = pt2
    dist = (x1 - x2)**2 + (y1 - y2)**2
    dist = np.sqrt(dist)
    return dist
    
def get_z_v2(pt1, pt2, pt1_hat, pt2_hat, h):
    '''
    pt1: x1,y1 = Point1 in image1
    pt2: x2,y2 = Point2 in image1
    pt1_hat: x1_hat,y1_hat = Point1 in image2
    pt2_hat: x2_hat,y2_hat = Point2 in image2
    '''
    print('Trying 2nd version of DEPTH CALCULATION')
    print('2nd Version pt1:', pt1)
    print('2nd Version pt2:', pt2)
    print('2nd Version pt1_hat:', pt1_hat)
    print('2nd Version pt2_hat:', pt2_hat)
    Z = (distance(pt1, pt2) * h)/distance(pt1_hat, pt2_hat)
    return Z

def generate_area(x1,y1, x2, y2, h, f1, f2, depth=0):
    # print('x1',x1)
    print('y1',y1)
    # print('x2',x2)
    print('y2',y2)

    X = np.zeros(4)
    Y = np.zeros(4)
    Z = get_z(y1[1],y2[1], h, f1, f2)
    print('Z:', Z)
    Z = get_z(y1[2],y2[2], h, f1, f2)
    print('Z:', Z)
    Z = get_z(y1[3],y2[3], h, f1, f2)
    print('Z:', Z)
    Z = get_z(y1[0],y2[0], h, f1, f2)
    print('Z:', Z)
    # print(x1)
    # print(x2)
    print(Z)
    # X = (Z*x1)/f1
    # Y = (Z*y1)/f1

    ##2ND version Depth Calculation
    pt1 = x1[0], y1[0]
    pt2 = x1[1], y1[1]
    pt1_hat = x2[0], y2[0]
    pt2_hat = x2[1], y2[1]
    print('2ndVersion:', get_z_v2(pt1_hat, pt2_hat, pt1, pt2, h))

    if depth !=0:
        X = (depth*x1)/f1
        Y = (depth*y1)/f1
    length = np.sqrt((X[0]-X[1])**2 + (Y[0]-Y[1])**2)
    breadth = np.sqrt((X[2]-X[1])**2 + (Y[2]-Y[1])**2)
    area = length*breadth
    return area 

def call_this(name, h, focalLength = 2.96347438e+03, depth =0):

    # name = 4
    path = './storedCoordinatesForDepthCalc/'

    # image_points = np.loadtxt('../rooftop/area_test/ud/'+str(name)+'.out', delimiter=',')
    image_points = np.loadtxt(path+str(name)+'.out', delimiter=',')
    image_points1 = np.array(image_points)

    # image_points = np.loadtxt('../rooftop/area_test/ud/'+str(name+1)+'.out', delimiter=',')
    image_points = np.loadtxt(path+str(name+1)+'.out', delimiter=',')
    image_points2 = np.array(image_points)
    
    cx = 2.20429432e+03
    cy = 1.47383511e+03
    print('image1 Points:', image_points1)
    print('image2 Points:', image_points2)
    # image_points1[:,0] = image_points1[:,0]  - cx
    # image_points1[:,1] = image_points1[:,1]  - cy
    # image_points2[:,0] = image_points2[:,0]  - cx
    # image_points2[:,1] = image_points2[:,1]  - cy
    # print(image_points1)

    f1 = focalLength
    f2 = f1
    # h = 7
    if depth != 0:
        area = generate_area(image_points1[:,0],image_points1[:,1], image_points2[:,0], image_points2[:,1], h, f1, f2, depth)
    else:
        area = generate_area(image_points1[:,0],image_points1[:,1], image_points2[:,0], image_points2[:,1], h, f1, f2)        
    print('area: ', area)

# call_this(1,1.2)
# call_this(3,2.6)
# call_this(4,7)

# call_this(3,20.2, 90.4)
# call_this(4,2.6, 93)


call_this(1, 1.2, focalLength, 69)
call_this(2, 20.2, focalLength, 70.2)
call_this(3, 2.6, focalLength, 90.4)
call_this(4,7, focalLength, 93)
# call_this(5,2.6, 100)

def calcNetArea(contours, hierarchy, depth, focalLength):
    sorted_contours = sorted(contours, key=cv2.contourArea)
    maxContour = sorted_contours[len(contours) - 1]
    maxCont = np.squeeze(maxContour, axis=1)
    hierarchy = np.squeeze(hierarchy, axis=0)

    maxCont = maxCont*(depth/focalLength)

    # outer_polygon_area = Polygon(maxCont).area*(depth/focalLength)**2
    outer_polygon_area = Polygon(maxCont).area
    child_polygon_totalarea = 0.0
    maxContourIndex = -1

    for i in range(len(contours)):
        if contours[i] is maxContour:
            maxContourIndex = i
            break
    index = maxContourIndex

    print(hierarchy[maxContourIndex])
    print('hierarchy:', hierarchy)

    if hierarchy[index][2] != -1: ## check if child is there or not
        for i in range(len(contours)):
            child_contour = contours[i]
            if  child_contour is maxContour:
                print('Reached max Contour')
                continue
            elif hierarchy[i][3] == maxContourIndex: ## Is a child
                print('hierarchy of child:', hierarchy[i])
                c = contours[i]
                c = np.squeeze(c, axis=1)
                c = c*(depth/focalLength)
                # child_polygon_area = Polygon(c).area*(depth/focalLength)**2
                child_polygon_area = Polygon(c).area
                child_polygon_totalarea += child_polygon_area
                print('childPolygonArea: ', child_polygon_area)
                print('child Total area in loop:', child_polygon_totalarea)
    else:
        print('No Child Polygon is there')
    net_area = outer_polygon_area - child_polygon_totalarea
    print('OUTER POLYGON AREA:', outer_polygon_area)
    print('CHILD TOTAL AREA:', child_polygon_totalarea)
    print('NET AREA:', net_area)
    return net_area
    
im = cv2.imread('.\\input_to_findArea\\gray_preprocessed_undistorted_avgPooling\\5.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray.astype(np.uint8), 200, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
print('No of contours:', len(contours))
# cv2.drawContours(im, contours, -1, (0,255,0), 3)
## Draw max area contour
c = max(contours, key = cv2.contourArea)
cv2.drawContours(im, [c], 0, (0,255,0), 3)
c = np.squeeze(c, axis=1)
print('Final Outer Polygon Area:', Polygon(c).area*(depth/focalLength)**2)    
# for pt in c:
#     print(pt)
plt.imshow(im)
plt.show()

calcNetArea(contours, hierarchy, depth, focalLength)

sorted_contours = sorted(contours, key=cv2.contourArea)
maxContour = sorted_contours[len(contours) - 1]
hierarchy = np.squeeze(hierarchy, axis=0)
maxContourIndex = -1
for i in range(len(contours)):
    if contours[i] is maxContour:
        maxContourIndex = i
        break
print(hierarchy[maxContourIndex])

## Draw all contours
img_allcontours = im.copy()
cv2.drawContours(img_allcontours, contours, -1, (0,0,255), 5)
plt.imshow(img_allcontours)
plt.show()
