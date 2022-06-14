import cv2
import numpy as np
from requests import get

def SelectPointsInImage(PathIn, Image=None):
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

	return Image, click_list

def display_selected_contour(image, click_list):
    
    # Displays the selected contour in an image
    # click_list: list of points selected in the image
    image1 = image
    for i in range(len(click_list)-1):
        cv2.line(image1, click_list[i], click_list[i+1], (0,0,255), 2)
    
    while True:
        cv2.imshow('Selected Contour', image1)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

def get_keypoints(pathin1, pathin2, imagestxt):

        file = open(imagestxt, 'r')
        image_name1 = pathin1.split('/')[-1]
        image_name2 = pathin2.split('/')[-1]
        # print(image_name1)
        # print(image_name2)
        #First 4 lines do not contain camera pose or 3d points information
        file_list = file.readlines()[4:]
        keypoints1 = []
        keypoints2 = []
        for i in range(0,len(file_list),2):

            file_list[i] = file_list[i].strip('\n')
            image_name = file_list[i].split(' ')[-1]
            
            if image_name1 == image_name:
                file_list[i+1] = file_list[i+1].strip('\n')
                keypoints1 = np.asarray(list(file_list[i+1].split(' '))).reshape(-1,3).astype(np.float64)
            if image_name2 == image_name:
                file_list[i+1] = file_list[i+1].strip('\n')
                keypoints2 = np.asarray(list(file_list[i+1].split(' '))).reshape(-1,3).astype(np.float64)
        
        reconstructed_keypoints1_indices = np.where(keypoints1[:,2]!=-1)
        reconstructed_keypoints1 = keypoints1[reconstructed_keypoints1_indices,:].reshape(-1, 3)
        
        reconstructed_keypoints2_indices = np.where(keypoints2[:,2]!=-1)
        reconstructed_keypoints2 = keypoints2[reconstructed_keypoints2_indices,:].reshape(-1, 3)

        return reconstructed_keypoints1, reconstructed_keypoints2

def get_bounds(keypoints, threshold=10):
    
    keypoints = np.asarray(keypoints).reshape(-1, 2)
    y_min = np.min(keypoints[:,0])
    y_max = np.max(keypoints[:,0])
    x_min = np.min(keypoints[:,1])
    x_max = np.max(keypoints[:,1])

    return [y_min-threshold, y_max+threshold, x_min-threshold, x_max+threshold]

def get_close_keypoint(bound, keypoints, threshold=5):

    # Returns the indices of the keypoints closer to the selected contour line
    # Threshold indicates the maximum perpendicular distance of the keypoint from the line in terms of pixels
    close_keypoints = []
    y_min, y_max, x_min, x_max = bound
    for i in range(len(keypoints)):
        y = keypoints[i,0]
        x = keypoints[i,1]
        if y_min<=y<=y_max and x_min<=x<=x_max:
            close_keypoints.append(keypoints[i,:])
    
    close_keypoints = np.asarray(close_keypoints).reshape(-1, 3)

    return close_keypoints

def plot_keypoints(image, keypoints):

    image1 = image 
    points = tuple(map(tuple, keypoints[:,0:2]))
    
    for i in range(len(points)):
        cv2.circle(image1, (int(points[i][0]),int(points[i][1])), 2, (0, 0, 255), 2)
    
    while True:
        cv2.imshow('Plotted Keypoints', image1)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

def get_point_indices(keypoints):

        #Input: keypoints array
        #Output: A list of point index for each image, the corresponding 3D points can be found out from points3d.txt using these indices

        return list(keypoints[:,2])

def get_3DPoints(point_indices, points3dtxt):
        
        #Input: List of point_indices, Points3d.txt file path that contains 3D points generated using colmap
        #Output: Array of 3D points, only the 3D points corresponding to the point_indices are returned

        file = open(points3dtxt, 'r')
        file_list = file.readlines()[3:]
        
        #Creating an array of all the 3d points
        points = []
        
        for i in range(len(file_list)):
            file_list[i] = file_list[i].strip('\n')
            line_list = file_list[i].split(' ')
            points.append([float(line_list[0]), float(line_list[1]), float(line_list[2]), float(line_list[3])])

        points = np.asarray(points).reshape(len(points),4)
        points = points[points[:, 0].argsort()]
    
        #Selecting only the 3D points corresponding to the point_indices
        contour_points = []
        for i in range(len(points)):
            if int(points[i,0]) in point_indices:
                contour_points.append([points[i,1], points[i,2], points[i,3]])
        
        contour_points = np.asarray(contour_points).reshape(-1,3)
        
        return contour_points

def get_mesh_distance(points3d1, points3d2):

    mean1 = np.mean(points3d1, axis=0)
    mean2 = np.mean(points3d2, axis=0)

    mesh_distance = ((mean1 - mean2)**2).sum()**0.5
    return mesh_distance

def draw_box(image, bounds):
    
    image1 = image
    y_min, y_max, x_min, x_max = bounds
    cv2.rectangle(image1, (y_min, x_min), (y_max, x_max), (0, 255, 255), 2)
    while True:
        cv2.imshow('Bound box', image1)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pathin1 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/images/18.jpg'
    pathin2 = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/images/26.jpg'
    imagestxt = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/sparse/0/images.txt'
    points3dtxt = '/home/kushagra/IIIT-H/FromTopVideos/DJI_0378/sparse/0/points3D.txt'
    image1 = cv2.imread(pathin1)
    image2 = cv2.imread(pathin2)

    #Asking the user to select line contours between which distance is to be found out
    image1, selected_points1 = SelectPointsInImage(pathin1)
    image2, selected_points2 = SelectPointsInImage(pathin2)
    
    #Displaying the selected contour lines
    display_selected_contour(image1, selected_points1)
    display_selected_contour(image2, selected_points2)

    #Finding out all the features involved in 3d reconstruction
    keypoints1, keypoints2 = get_keypoints(pathin1, pathin2, imagestxt)

    bound1 = get_bounds(selected_points1)
    bound2 = get_bounds(selected_points2)

    draw_box(image1, bound1)
    draw_box(image2, bound2)
   
    keypoints1, keypoints2 = get_keypoints(pathin1, pathin2, imagestxt)
    close_keypoints1 = get_close_keypoint(bound1, keypoints1)
    # print(close_keypoints1.shape)
    close_keypoints2 = get_close_keypoint(bound2, keypoints2)    
    # print(close_keypoints2.shape)

    # print(close_keypoints1.shape)
    # print(keypoints1.shape)
    # print(close_keypoints1[0:10,:])
    close_keypoints1_indices = get_point_indices(close_keypoints1)
    close_keypoints2_indices = get_point_indices(close_keypoints2)
    close_3dpoints1 = get_3DPoints(close_keypoints1_indices, points3dtxt)
    close_3dpoints2 = get_3DPoints(close_keypoints2_indices, points3dtxt)
    print(close_3dpoints1)
    print(close_3dpoints2)
    print(get_mesh_distance(close_3dpoints1, close_3dpoints2)*0.6305697493044669)
    # plot_keypoints(image1, keypoints1)
    plot_keypoints(image1, close_keypoints1)
    plot_keypoints(image2, close_keypoints2)

    

#inside edges = 33m
#outside edges = 56.34m 
#left outside to right inside = 44.77m
#right outside to left inside = 45.66m
#outside edge distance = 56.39m
#scale = 11.969790637731599