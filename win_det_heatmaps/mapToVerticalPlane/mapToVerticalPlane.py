#!/usr/bin/env python
# coding: utf-8

import os
from re import M
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from pandas import *

class MapToVerticalPlane():
    def __init__(self, focalLength):
        '''
        INPUT -
        focalLength: focal length(in pixels) of UAV camera

        imgSeqPath: Dir Path to all images and log of the sequence
        coordFilePath: Path to CSV File where window coordinates in each image of the seq are stored 
        windowCount: global Window Count in the Image sequence
        storeyCount: global story Count in the Image sequence
        nmsThresh: threshold for Non-maxima suppression
        final_building_boxes: Final coordinates of all windows of the sequence mapped on the vertical Plane
        '''
        
        print('MapToVerticalPlane() ')
        self.imgSeqPath = None
        self.coordFilePath = None
        self.depth = None
        self.focalLength = focalLength
        self.windowCount = 0
        self.storeyCount = 0
        self.nmsThresh = 0
        self.final_building_boxes = np.array([])
        self.imgShape = None


    def loadCoordsFromCSV(self, csvFilePath):
        '''
        Data Loader for CSV Logs and Mapping Input to 4 window coords 
        Loading Coords from CSV File & Mapping to all 4 coords of Windows logic
        '''
        with open(csvFilePath, newline='') as f:
            csvread = csv.reader(f)
            batch_data = list(csvread)

        batch_data_int = []
        for inner_list in batch_data:
            innet_out_list = []
            for string in inner_list:
                if string == 'nan':
                    string = np.nan
                innet_out_list.append((float(string)))
            batch_data_int.append(innet_out_list)

        FinalList = []
        for i in range(len(batch_data_int)):
            el = batch_data_int[i]
            elChunks = [el[x:x+4] for x in range(0, len(el), 4)]
            newElChunks = elChunks.copy()
            for i in range(len(elChunks)):
                chunk = elChunks[i]
                newChunk = chunk.copy()

                newChunk.insert(2, chunk[0])
                newChunk.insert(3, chunk[3])
                newChunk.insert(6, chunk[2])
                newChunk.insert(7, chunk[1])
                newElChunks[i] = newChunk
            perImageCoords = np.array(newElChunks)
            perImageCoords = perImageCoords.reshape(-1,4,2)
            FinalList.append(perImageCoords)

        # print("Final list size:", len(FinalList))
        return FinalList


    def mapToAll4Coords(self, input):
        '''
        Here, input is in form of list of sX, sY, eX, eY i.e. only top-left and bottom-right corner coordinates
        INPUT -
        input: [[[s1x_1,s1y_1,e1x_1,e1y_1], [s2x_1, s2y_1, e2x_1, e2y_1]],   [[s1x_2,s1y_2,e1x_2,e1y_2],[s2x_2,s2y_2,e2x_2,e2y_2]]]
        
        Here NOTE: 1st element i.e. [[s1x_1,s1y_1, e1x_1,e1y_1], [s2x_1, s2y_1, e2x_1, e2y_1]] corresponds to 1st image
        [s1x_1, s1y_1, e1x_1, e1y_1] -> 1st window in Image1
        [s2x_1, s2y_1, e2x_1, e2y_1] -> 2nd window in Image1
        '''
        inputInListOfLists = [arr.tolist() for arr in input]
        finalListMappedWithAll4Coords = []
        for image in inputInListOfLists:
            finalSubList = []
            imageClone = image.copy()
            for i in range(len(image)):
                window = image[i]
                windowClone = window.copy()
                windowClone.insert(2, window[0])
                windowClone.insert(3, window[3])
                windowClone.insert(6, window[2])
                windowClone.insert(7, window[1])
                imageClone[i] = windowClone
            perImageCoords = np.array(imageClone)
            perImageCoords = perImageCoords.reshape(-1,4,2)
            finalListMappedWithAll4Coords.append(perImageCoords)
        
        return finalListMappedWithAll4Coords

    ## Utils
    def calculateRange(self, coordinates, padding):
        '''
        Adds some padding to top-left and bottom-right coordinates of window corner.
        This is because the model inference might produce tight bounding boxes so in order to avoid cropping window corners, 
        we add this padding.

        INPUT - 
        coordinates: input coordinates of a single window
        padding: no. of pixels added as padding

        OUTPUT -
        startX, endX, startY, endY: start and end corners of window after padding
        '''
        minX = maxX = coordinates[0][0]
        minY = maxY = coordinates[0][1]
        startX = startY = endX = endY = 0
        h,w,c = self.imgShape
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
        return startX, endX, startY, endY

    def prepareBinaryMask(self, img_gray, coordinates, padding = 5):
        ret, binary_img = cv2.threshold(img_gray, 255, 255, cv2.THRESH_BINARY)
        for i in range(coordinates.shape[0]):
            startX, endX, startY, endY = self.calculateRange(coordinates[i], padding)
            binary_img[startY:endY, startX:endX] = 1
        return binary_img

    def addPaddingToWindowCoords(self, coordinates, padding = 5):
        '''
        Iterates over all coords and adds some padding. This is because the model inference might have tight bounding boxes.
        '''
        pick = []
        for i in range(coordinates.shape[0]):
            startX, endX, startY, endY = self.calculateRange(coordinates[i], padding)
            pick.append([startX, startY, endX, endY])
        return pick

    def plotBoxes(self, verticalPlane, boxes):
        for box in boxes:
            sX, sY, eX, eY = box
            cv2.rectangle(verticalPlane, (int(sX), int(sY)), (int(eX), int(eY)), (255, 255, 255), 3)
        
        fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(25, 500))
        ax1.imshow(verticalPlane, cmap="gray")
        ax1.set_xlabel("PLANE", fontsize=14)
        plt.show()
        cv2.imwrite('after_nms_vertical_plane.png',verticalPlane)

    ## Plane Mapping, NMS and Storey Logic Functions

    # FOV = 82.6 degrees, imgDim = 720*960, FocalLength = 410 pixels
    def mapToVerticalPlane(self, verticalPlane, img, boundingBoxes, height, y_correction, seq_num):
        '''
        Maps windows in an image on a vertical Plane.
        
        INPUT - 
        verticalPlane: dummy black plane where all windows would be visualized
        img: img in a sequence
        boundingBoxes:  all window coordinates of an img
        height: Height of UAV(in cm)
        y_correction: Correcton in Y-axis due to orientation error
        seq_num: seq no. Is used to visualize all sequences parallely(with some distance approx. equal to w[width of image] between 2 sequences) on the vertical plane.
        
        OUTPUT - 
        Returns mapped Bounding boxes of a single image on the vertical Plane
        '''
        mappedBoundingBoxes = []
        heightOfPlane = verticalPlane.shape[0]
        h, w, c = self.imgShape
        
        # Visualize each seq at a distance of w width
        seq_space = w
        
        for box in boundingBoxes:
            sX, sY, eX, eY = box
            # print(box)
            
            sY = h/2 - sY
            eY = h/2 - eY
            
            mappedSX, mappedSY, mappedEX, mappedEY = sX + (seq_num - 1)*seq_space, int(heightOfPlane - ((sY*self.depth/self.focalLength) + y_correction + height)), eX + (seq_num - 1)*seq_space, int(heightOfPlane - ((eY*self.depth/self.focalLength) + y_correction + height))            
            
            cv2.rectangle(verticalPlane, (mappedSX, mappedSY), (mappedEX, mappedEY), (255, 255, 255), 4)
            
            mappedBoundingBoxes.append((mappedSX, mappedSY, mappedEX, mappedEY))
        return mappedBoundingBoxes


    def non_max_suppression_fast(self, boxes, overlapThresh):
        '''
        NMS - Non-maxima suppression
        
        INPUT- 
        boxes: input window coordinates
        overlapThresh: Threshold for suppressing window bounding boxes 

        OUTPUT-
        Returns unique bounding boxes
        '''
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
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
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
            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def calculateStoreys(self, coords, heightOfPlane = 1500):
        '''
        Calculates storeys from coordinates of all detected windows
        '''
        yTop = coords[:,1]
        yBottom = coords[:,3]
        yAvg = (yTop + yBottom)/2
        storeyCount = 1 if len(coords) > 0 else 0
        index = 0
        storeyHeights = [heightOfPlane - yAvg[0]] if len(coords) > 0 else []
        '''
        print('yTop:', yTop)
        print('yAvg:', yAvg)
        print('yBottom:', yBottom)    
        '''
        print('yAvg Heights:', heightOfPlane - yAvg)

        singleStoreyHeight = []
        avgStoreyHeights = []
        
        for i in range(len(coords)):
            if (((yAvg[index] > yTop[i]) and (yAvg[index] < yBottom[i])) or 
                ((yTop[index] > yTop[i]) and (yTop[index] < yBottom[i]))  or
                ((yBottom[index] > yTop[i]) and (yBottom[index] < yBottom[i]))):
                singleStoreyHeight.append(heightOfPlane - yAvg[i])
                continue
            else:
                avgStoreyHeights.append(sum(singleStoreyHeight)/len(singleStoreyHeight))
                singleStoreyHeight = []
                index = i
                storeyCounted = False
                for j in range(i):  
                    if (((yAvg[index] > yTop[j]) and (yAvg[index] < yBottom[j])) or 
                        ((yTop[index] > yTop[j]) and (yTop[index] < yBottom[j])) or
                        ((yBottom[index] > yTop[j]) and (yBottom[index] < yBottom[j]))):
                        storeyCounted = True
                        break
                if(storeyCounted == False):
                    storeyHeights.append(heightOfPlane - yAvg[index])
                    singleStoreyHeight.append(heightOfPlane - yAvg[i])
                    storeyCount+=1
                    
        avgStoreyHeights.append(sum(singleStoreyHeight)/len(singleStoreyHeight))
        print('avgStoreyHeights: ', avgStoreyHeights)
        print("storeyHeights:", storeyHeights)
        print('StoreyCount after running post processing module: ', storeyCount)
        return storeyCount, storeyHeights, avgStoreyHeights


    ## IMU Functions
    def readSeqLogs(self, seqLogFile):
        '''
        Reads Logs info(imageName, height info, orientation info[roll, pitch, yaw] etc) from a text file of a specific Sequences
        NOTE: The text file has a specific format. So you need to input logs in a specific format only.
        Sample logs file is provided in the same dir.
        '''
        data = read_csv(seqLogFile, sep = " , ", header=None) #read imu data file
        #print(data[1][0][1:-1].split(',')[9].split(':')[1])

        imgNames = [(data[0][i].split(': ')[1]) for i in range(len(data))]
        print(imgNames)

        heights_allImages = [int(data[1][i][1:-1].split(',')[9].split(':')[1]) for i in range(len(data))]
        print('heights: ', heights_allImages)

        # in degrees
        pitch_allImages = [int(data[1][i][1:-1].split(',')[0].split(':')[1]) for i in range(len(data))]
        print('pitch: ', pitch_allImages)

        # in degrees read roll and yaw
        roll_allImages = [int(data[1][i][1:-1].split(',')[1].split(':')[1]) for i in range(len(data))]
        print('roll: ', roll_allImages)

        yaw_allImages = [int(data[1][i][1:-1].split(',')[2].split(':')[1]) for i in range(len(data))]
        print('yaw: ', yaw_allImages)

        #imu correction for pitch
        y_corrections = [-self.depth*np.tan(np.radians(int(pitch_allImages[i]))) for i in range(len(pitch_allImages))]
        print('y_corrections for pitch: ', y_corrections)
        
        orientation_allImages = [roll_allImages, pitch_allImages, yaw_allImages]
        return imgNames, heights_allImages, orientation_allImages, y_corrections


    ## Running Vertical Plane Mapping
    def runVerticalMap(self, verticalPlane, imgSeqPath, coordFilePath, depth, offset_ramp = 0, seq_num = 1):
        '''
        Runs Vertical Plane Mapping Algorithm on entire image Sequence
        
        INPUT-
            verticalPlane: dummy vertical Plane where windows are visualized
            imgSeqPath: Dir where all images of a sequence and their log file is there
            coordFilePath: Path to CSV file containing window coords in all images generated by model inference and post-processing
            depth: Depth(in cm) of UAV to the building
            offset_ramp: Height above the ground(in cms) from where the UAV takes off for capturing the sequence. Usually it is 0, however comes in when UAV takes off from slope
            seq_num: seq no. Eg: building has multiple faces. so each vertical run is considered as seq. And such seq starts from 1. 
        
        OUTPUT-
            windowCount: Total window count in the sequence
            storeyCount: Storey Count in the sequence
            avgStoreyHeights: Avg Storey Heights in the sequence
        '''
        ## Loading coords from CSV File
        self.depth = depth
        FinalList = self.loadCoordsFromCSV(coordFilePath)
        seqLogFile = os.path.join(imgSeqPath, 'myfile.txt')
        imgNames, heights_allImages, orientation_allImages, y_corrections = self.readSeqLogs(seqLogFile)    

        # grab the paths to the input images and initialize our images list
        
        print("[INFO] loading images...")
        ### OLD DATA
        # imagePaths = sorted(list(paths.list_images(imgSeqPath)))
        imagePaths = [os.path.join(imgSeqPath, imgName) for imgName in imgNames]
        # print(imagePaths)

        assert len(heights_allImages) == len(imagePaths), "Height Info and ImageSeq does not match. Please provide correct height info for seq"
        assert len(FinalList) == len(imagePaths), "Window coords CSV File and ImageSeq does not match."
        images = []

        # loop over the image paths, load each one, and add them to our
        for imagePath in imagePaths:
            image = cv2.imread(imagePath, 1)
            images.append(image)
            print(imagePath)

        ## Image shape 
        self.imgShape = images[0].shape

        windowCount = len(FinalList[0])
        print("startWindowCount:", windowCount)
        allMappedBoxes = []
        ## MAIN LOOP ##
        for i in range(len(images)):
            img1 = images[i]

            # read images and transform them to grayscale
            # Make sure that the train image is the image that will be transformed
            # Opencv defines the color channel in the order BGR 
            # Transform it to RGB to be compatible to matplotlib
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            
            '''
            fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(16,9))
            ax1.imshow(img1, cmap="gray")
            ax1.set_xlabel("img1", fontsize=14)
            plt.show()
            '''

            ## Preparing Binary Mask and mapping bounding box coordinates
            ## COORDS1: WOULD COME FROM TEXT FILEW
            coords1 = FinalList[i]

            ## CHECK NAN Condition if so skip as there are no windows in the img
            if coords1.shape[0] == 1 and np.any(np.isnan(coords1)): 
                print('Skipping as there is no window')
                continue

            coords1 = coords1.astype(int)
            padding = 0

            ## Commented as the coords would come from recorded text File.
            pick1 = self.addPaddingToWindowCoords(coords1, padding)

            mappedBoxes = self.mapToVerticalPlane(np.copy(verticalPlane), img1, pick1, heights_allImages[i], y_corrections[i], seq_num)
                        
            allMappedBoxes = allMappedBoxes + mappedBoxes
            allMappedBoxes = self.non_max_suppression_fast(np.array(allMappedBoxes), self.nmsThresh).tolist()

            '''
            fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(25, 500))
            ax1.imshow(verticalPlaneCopy, cmap="gray")
            ax1.set_xlabel("PLANE", fontsize=14)
            plt.show()
            '''

        ## NMS
        # print("before NMS:", np.array(allMappedBoxes))
        # self.plotBoxes(np.copy(verticalPlane), allMappedBoxes)
        # print(np.array(allMappedBoxes).shape)

        finalBoxes = self.non_max_suppression_fast(np.array(allMappedBoxes), self.nmsThresh)
        #finalBoxes = np.array(finalBoxes)
        print("after NMS finalBoxes:", finalBoxes)
        print('finalBoxes shape', finalBoxes.shape)
        
        if seq_num == 1:
            self.final_building_boxes = finalBoxes
        else:
            self.final_building_boxes = np.concatenate((self.final_building_boxes,finalBoxes))
        
        self.plotBoxes(np.copy(verticalPlane), finalBoxes)
        storeyCount, storeyHeights,avgStoreyHeights = self.calculateStoreys(finalBoxes, heightOfPlane = verticalPlane.shape[0])

        ## Final Result of Vertical Plane Mapping Approach 
        
        windowCount = finalBoxes.shape[0]
        self.windowCount += windowCount
        print("Total no. of windows in seq: " + str(windowCount))
        print("StoreyCount:", storeyCount)

        print("FinalBoxes of seq(in list):", list(finalBoxes))

        # Height Differences between stories of a single vertical sequence
        print('Diff in storeyheights:', np.diff(np.array(storeyHeights)))
        print('Diff in AvgStoreyHeights:', np.diff(np.array(avgStoreyHeights)))

        # apply offset using imu data of the first image captured
        # offset is due to the ground ground visibility in the first capture
        # --- Assuming the the camera is orthogonal and in level with the window 
        # --- (window centre and camera centre coinciding)

        offset_building = avgStoreyHeights[0] - heights_allImages[0]

        offset_final = offset_building - offset_ramp
        print('Offset Final:', offset_final)
        storeyHeights_final = avgStoreyHeights - offset_final*(np.array(np.ones(len(avgStoreyHeights))))
        print('final storey heights', storeyHeights_final)

        print('Height difference between 2 consecutive stories of a single vertical sequence', 
            np.diff(np.array(storeyHeights_final)))

        # imu plot data for this seq
        '''
        plt.figure()
        plt.plot(heights_allImages, label = 'IMU All Image Height Data')
        plt.plot(storeyHeights_final, label = 'Storey Heights of a ' + str(storeyCount) + ' storey sequence')
        t_line = range(len(heights_allImages))
        for i in range(len(storeyHeights_final)):
            plt.plot(t_line, np.repeat(storeyHeights_final[i], len(heights_allImages)),
                    label = str(len(storeyHeights_final) - i - 1) + ' storey avg height') 
        plt.legend(bbox_to_anchor=(2, 1))
        plt.title('Average or Mid heights of stories')
        plt.show()
        '''

        return self.final_building_boxes, windowCount, storeyCount, avgStoreyHeights

if __name__ == '__main__':
    coord_dir = "../sample_seq_data"

    imgPath = "../sample_seq_data/001_new/images"
    coordFilePath = os.path.join(coord_dir, 'coordinatesFromPostProcessing-1_new-shufflenet.csv')

    ##  TESTING ONLY SINGLE SEQUENCE
    imgPath2 = "../sample_seq_data/002_new/images"
    coordFilePath2 = os.path.join(coord_dir, 'coordinatesFromPostProcessing-2_new-shufflenet.csv')

    imgPath3 = "../sample_seq_data/003_new/images"
    coordFilePath3 = os.path.join(coord_dir, 'coordinatesFromPostProcessing-3_new-shufflenet.csv')

    verticalPlane = np.zeros((1500,10000,3),np.uint8)

    depth = 750 ## Depth to the building(in cm)

    offset_ramp_001 = 15 #cm for seq 001
    offset_ramp_002 = 81 #cm for seq 002
    offset_ramp_003 = 0 #cm for seq 003

    mapToVerticalPlane = MapToVerticalPlane(focalLength = 920)

    seq_num = 1
    final_building_boxes, windowCount1, storeyCount1, avgStoreyHeights1 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath, coordFilePath, depth, offset_ramp_001, seq_num)
    seq_num = 2
    final_building_boxes, windowCount2, storeyCount2, avgStoreyHeights2 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath2, coordFilePath2, depth, offset_ramp_002, seq_num)
    seq_num = 3
    final_building_boxes, windowCount3, storeyCount3, avgStoreyHeights3 = mapToVerticalPlane.runVerticalMap(verticalPlane, imgPath3, coordFilePath3, depth, offset_ramp_003, seq_num)
    
    mapToVerticalPlane.plotBoxes(np.copy(verticalPlane), final_building_boxes)

    print('Total window count of all seqs:', str(mapToVerticalPlane.windowCount))


