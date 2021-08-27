#!/usr/bin/env python
import cv2   
import numpy as np   
import matplotlib.pyplot as plt
import imutils
from scipy.spatial import ConvexHull
import logging

logging.basicConfig(filename='postProcess.log', format='%(levelname)s: %(message)s',  level=logging.INFO)

class PostProcess():
    def __init__(self, img, group_corners_wz_score):
        logging.info('Running postProcessing module')
        self.img = img
        ## Create copies of image for plotting purpose.
        self.img_rgb = np.copy(img)
        self.img_show = np.copy(img)
        self.img_copy = np.copy(img)
        self.img_copy_new = np.copy(img)
        self.mappedInputArr = None
        self.mapInputForPostProcessing(group_corners_wz_score)
        self.window_count = 0
        self.storey_count = 0

    def mapInputForPostProcessing(self, group_corners_wz_score):
        mappedInput = []
        for window in group_corners_wz_score:
            # print(window['position'])
            # print(window['position'][0])
            for windowCoords in window['position']:
                mappedInput.append([windowCoords[0], windowCoords[1]])

        mappedInputArr = np.array(mappedInput)
        self.mappedInputArr = np.reshape(mappedInputArr,(mappedInputArr.shape[0]//4 , 4, mappedInputArr.shape[1]))

    def calculateRange(self, coordinates, padding = 10):
        minX = maxX = coordinates[0][0]
        minY = maxY = coordinates[0][1]
        startX = startY = endX = endY = 0
        h,w,c = self.img_rgb.shape
        for i in range(len(coordinates)):
            if minX > coordinates[i][0]:
                minX = coordinates[i][0]
            if maxX < coordinates[i][0]:
                maxX = coordinates[i][0]

            if minY > coordinates[i][1]:
                minY = coordinates[i][1]
            if maxY < coordinates[i][1]:
                maxY = coordinates[i][1]
        
        if (minX - padding < 0):
            startX = 0
        else:
            startX = minX - padding
        if (minY - padding < 0):
            startY = 0
        else:
            startY = minY - padding
        if (maxX + padding >= w):
            endX = w - 1
        else:
            endX = maxX + padding
        if (maxY + padding >= h):
            endY = h - 1
        else:
            endY = maxY + padding
        return int(startX), int(endX), int(startY), int(endY)
    
    def nms_local(self, rects_all, res_scores_all, flag, overlapThresh = 0.25):
        boxes = np.array(rects_all)
        res_scores_all = np.array(res_scores_all)

        xCoords = boxes[:,0]
        yCoords = boxes[:,1]
        xCoords2 = boxes[:,2]
        yCoords2 = boxes[:,3]

        boxes_tuples = []
        #loop over the starting (x, y)-coordinates again
        for (x, y, x2, y2) in zip(xCoords, yCoords, xCoords2, yCoords2):
            # update our list of rectangles
            boxes_tuples.append((x, y, x2, y2))

        #print(boxes_tuples)

        # apply non-maxima suppression to the rectangles
        pick, resScores = self.non_max_suppression_fast(np.array(boxes_tuples), overlapThresh, res_scores_all, flag)
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))
        # loop over the final bounding boxes
        print(pick)
        for (sX, sY, eX, eY) in pick:
            # draw the bounding box on the image
            cv2.rectangle(self.img_copy_new, (sX, sY), (eX, eY),
                (255, 0, 0), 3)
        plt.figure(num = 'nms')
        plt.imshow(self.img_copy_new)
        plt.title("Image After Local NMS")
        plt.show()
        return pick, resScores
    
    def matchTemplate(self, coordinates, fileName, padding=10):    

        startX, endX, startY, endY = self.calculateRange(coordinates, padding)
        plt.imshow(self.img_rgb)
        # print(startX, endX, startY, endY)
        #print(img_rgb.shape)
        
        template = self.img_rgb[startY:endY, startX:endX, :]
        searchImg = self.img_rgb[startY:endY, : ,:]
        searchImg_copy = np.copy(searchImg)

        self.template_plot(template,searchImg,fileName,fileName)
        
        img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_BGR2GRAY)   
        searchImg_gray = cv2.cvtColor(searchImg, cv2.COLOR_BGR2GRAY)
        
        rects = []
        res_scores = []
        for i in [-2.5, 0, 2.5]:

            ## DEBUG ##
            plt.figure(num= 'DEBUG!!')
            plt.imshow(searchImg)  
            plt.title('dEBUG search Image')
            plt.show()
            
            # Apply rotation and shear to template
            rotatedTemplate = imutils.rotate(template, i)
            plt.imshow(rotatedTemplate)
            plt.title('rotatedTemplate_' + str(fileName)) 
            plt.show()
            
            template_gray = cv2.cvtColor(rotatedTemplate, cv2.COLOR_BGR2GRAY)   

            # Store width in variable w and height in variable h of template  
            tW, tH = template_gray.shape[::-1]   
            # Now we perform match operations.   
            res = cv2.matchTemplate(searchImg_gray,template_gray,cv2.TM_CCOEFF_NORMED, 10)   
            # Declare a threshold   
            threshold = 0.550
            # Store the coordinates of matched region in a numpy array   
            loc = np.where( res >= threshold)
            logging.debug(loc)
            # Draw a rectangle around the matched region.   
            for pt in zip(*loc[::-1]): 
                #print('x: ' + str(pt[0]) + ' y: ' + str(pt[1]))
                cv2.rectangle(searchImg_copy, pt, (pt[0] + tW, pt[1] + tH), (255,0,0), 1)   
                rects.append((pt[0], pt[1], pt[0] + tW, pt[1] + tH))
                res_scores.append(res[pt[1]][pt[0]])

        # Now display the final matched template image
        plt.figure(num= 'SearchImage')
        plt.imshow(searchImg_copy)  
        plt.title('Result on search Image')
        plt.show()

        #cv2.imwrite('Template_' +  '0.6_threshold_' + fileName, template)
        #cv2.imwrite('TemplateMatched_'  + '0.6_threshold_' + fileName, searchImg)

        searchImgCoords = startX, endX, startY, endY
        logging.debug("rects: " + str(rects) + "\n")
        mappedCoords = self.mapCoordsToOriginalFrame(rects, searchImgCoords)
        logging.debug("mappedCoords: " + str(mappedCoords))

        for pt in (mappedCoords): 
            #print(str(pt[0]) + " "  + str(pt[1]))
            #print(str(pt[2]) + " "  + str(pt[3]))
            cv2.rectangle(self.img_show, (pt[0],pt[1]),  (pt[2], pt[3]), (255,0,0), 1)   
        
        # Now display the final matched template image   
        plt.figure(num = 'mapped' + fileName, figsize=(20,10))
        plt.imshow(self.img_show)  
        plt.title('Image - All Template Matching Result')
        
        coord = (coordinates[0][0],coordinates[0][1],coordinates[2][0],coordinates[2][1])
        mapped_new = mappedCoords + [coord]
        res_scores_new = res_scores + [10.0]
        
        mappedrects_picked, res_scores_picked = self.nms_local(mapped_new, res_scores_new, 1)
        
        return mappedrects_picked, res_scores_picked

    def non_max_suppression_fast(self, boxes, overlapThresh, res_scores_all, flag):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []
        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        # initialize the list of picked indexes	
        pick = []
        # grab the coordinates of the bounding boxes
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        #xxxx boxes by the bottom-right y-coordinate of the bounding box

        # sorting based on the score of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if flag == 1:
            idxs = np.lexsort((y2, res_scores_all))
        else:
            idxs = np.lexsort((res_scores_all, y2))

        #if res_scores_all[idxs[-1]] == 1.0:
        #    print('Model Detected')
        #    return 0,0

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            print('last box', boxes[i])
            pick.append(i)
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # print('w',w)
            # print('h',h)
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # print('overlap', overlap)
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int"), res_scores_all[pick]
    

    def mapCoordsToOriginalFrame(self, coords, searchImgCoords):
        mappedCoords = []
        startX, endX, startY, endY = searchImgCoords
        
        for i in range(len(coords)):
            mappedCoords.append((coords[i][0], coords[i][1] + startY, coords[i][2], coords[i][3] + startY))
        return mappedCoords

        ############### Calculating storey #####################
    def calculateStoreys(self, coords):
        yTop = coords[:,1]
        yBottom = coords[:,3]
        yAvg = (yTop + yBottom)/2
        print(yTop)
        print(yBottom)
        print(yAvg)
        storeyCount = 1 if len(coords) > 0 else 0
        index = 0
        for i in range(len(coords)):
            if ((yAvg[index] > yTop[i]) and (yAvg[index] < yBottom[i])):
                continue
            else:
                index = i
                storeyCounted = False
                for j in range(i):
                    if ((yAvg[index] > yTop[j]) and (yAvg[index] < yBottom[j])):
                        storeyCounted = True
                        break
                if(storeyCounted == False):
                    storeyCount+=1
        print('Number of storeys after running post processing module: ', storeyCount)
        self.storey_count = storeyCount
        return storeyCount

    def template_plot(self, template,searchImg,fileName,figname):
        plt.figure(num=figname, figsize=(20,10))
        plt.subplot(1, 2, 2)
        plt.imshow(template)
        plt.title('template_' + str(fileName))     
        plt.subplot(1, 2, 1)
        plt.imshow(searchImg)
        plt.title('templateMatched_' + str(fileName)) 
        plt.show()

    
    def runPostProcessingModule(self):

        mappedrects_picked_list = []
        res_scores_picked_list = []
        mappedrects_picked_list = np.array(mappedrects_picked_list)
        res_scores_picked_list = np.array(res_scores_picked_list)

        for ind in range(self.mappedInputArr.shape[0]):
            mappedrects_picked, res_scores_picked = self.matchTemplate(self.mappedInputArr[ind], ' ', padding=10)
            if ind == 0:
                mappedrects_picked_list = mappedrects_picked
                #res_scores_picked_list = np.expand_dims(res_scores_picked,0)
                res_scores_picked_list = res_scores_picked
            elif ind != 0:        
                mappedrects_picked_list = np.vstack((mappedrects_picked, mappedrects_picked_list))
                res_scores_picked_list = np.hstack((res_scores_picked, res_scores_picked_list))

            print('mappedrects_picked',mappedrects_picked_list)
            print('res_scores_picked', res_scores_picked_list)
            
        #---------NMS after TM of all model-detected windows----------

        #boxes is an array of bounding boxes each with 2 co-ordinates - (x1, y1, x2, y2)
        #lower left is (x1,y1), upper right is (x2,y2)
        
        overlapThresh = 0.25
        boxes = np.array(mappedrects_picked_list)
        res_scores_all = np.array(res_scores_picked_list)

        print(boxes)
        print(res_scores_all)
        print(boxes.shape)
        print(res_scores_all.shape)

        xCoords = boxes[:,0]
        yCoords = boxes[:,1]
        xCoords2 = boxes[:,2]
        yCoords2 = boxes[:,3]

        boxes_tuples = []
        #loop over the starting (x, y)-coordinates again
        for (x, y, x2, y2) in zip(xCoords, yCoords, xCoords2, yCoords2):
            # update our list of rectangles
            boxes_tuples.append((x, y, x2, y2))

        #print(boxes_tuples)
        logging.debug(boxes_tuples)

        # apply non-maxima suppression to the rectangles
        pick, resScores = self.non_max_suppression_fast(np.array(boxes_tuples), overlapThresh, res_scores_all, 0)
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))
        # loop over the final bounding boxes
        logging.info('final Bounding boxes after PostProcess: %s', (pick))
        for (sX, sY, eX, eY) in pick:
            # draw the bounding box on the image
            cv2.rectangle(self.img_copy, (sX, sY), (eX, eY),
                (255, 0, 0), 3)
        plt.figure(num = 'nms')
        plt.imshow(self.img_copy)
        plt.title("After NMS")

        cv2.imwrite('postProcess.png', self.img_copy)
        self.window_count = len(pick)
        logging.info('Number of windows detected by the model and post processing: %s', self.window_count)
        self.calculateStoreys(pick)
        
        
        ###### Display final result ###########
        plt.imshow(self.img_copy)
        plt.text(600, 700, "No. of window: " + str(self.window_count), color="blue", fontdict={"fontsize":10, "fontweight":'bold', "ha":"left", "va":"baseline"})
        plt.text(600, 650, "No. of storeys: " + str(self.storey_count), color="blue", fontdict={"fontsize":10, "fontweight":'bold', "ha":"left", "va":"baseline"})
        #Display the image
        plt.show()


