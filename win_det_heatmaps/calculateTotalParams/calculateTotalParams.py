import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from imutils import paths
from matplotlib import pyplot

class CalculateTotalParams():
    def __init__(self, recordedCoordsOfEntireSeq, height_info, imgShape = (720,960,3), focalLength = 920, depth = 700, 
        nmsThresh = 0, verticalPlaneShape = (1500, 1000, 3)):
        
        self.verticalPlane = np.zeros(verticalPlaneShape, np.uint8)
        self.imgShape = (720,960,3)
        self.height_info = height_info
        self.focalLength = 920 # in pixels
        self.depth = 700 # in cm
        self.nmsThresh = 0
        self.recordedCoordsOfEntireSeq = recordedCoordsOfEntireSeq ## in form of list of [[[sx,sy,ex,ey]]]

    def calculateRange(self, coordinates, padding):
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

    def mapToAll4Coords(self, input):
        ## Here input is in form of list of sX, sY, eX, eY
        ## Eg: input = [arr([[s1,s2, e1,e2], [s1, s2, e1, e2]]),   arr([[s3,s4,e3,e4],[s3,s4,e3,e4]])]
        inputInListOfLists = [arr.tolist() for arr in input]
        finalListMappedWithAll4Coords = []
        for imageCoords in inputInListOfLists:
            finalSubList = []
            imageClone = imageCoords.copy()
            for i in range(len(imageCoords)):
                window = imageCoords[i]
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

    # FOV = 82.6 degrees, imgDim = 720*960, FocalLength = 410 pixels
    def mapToVerticalPlane(self, boundingBoxes, height, verticalPlane):
        mappedBoundingBoxes = []
        heightOfPlane = verticalPlane.shape[0]
        h, w, c = self.imgShape
        for box in boundingBoxes:
            sX, sY, eX, eY = box
            print(box)
            
            sY = h/2 - sY
            eY = h/2 - eY
            mappedSX, mappedSY, mappedEX, mappedEY = sX, int(heightOfPlane - ((sY*self.depth/self.focalLength) + height)), eX, int(heightOfPlane - ((eY*self.depth/self.focalLength) + height))
            
            print("mapped:" + str(mappedSX) + " " +  str(mappedSY)  + " " + str(mappedEX) + " " + str(mappedEY))
            cv2.rectangle(verticalPlane, (mappedSX, mappedSY), (mappedEX, mappedEY), (255, 255, 255), 4)
            
            mappedBoundingBoxes.append((mappedSX, mappedSY, mappedEX, mappedEY))
        return mappedBoundingBoxes
            

    def non_max_suppression_fast(self, boxes):
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
    #         print("overlap: ", overlap)
            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > self.nmsThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    ############### Calculating storey & Storey Heights#####################
    def calculateStoreys(self, coords):
        yTop = coords[:,1]
        yBottom = coords[:,3]
        yAvg = (yTop + yBottom)/2
        storeyCount = 1 if len(coords) > 0 else 0
        index = 0
        heightOfPlane = self.verticalPlane.shape[0]
        storeyHeights = [heightOfPlane - yAvg[0]] if len(coords) > 0 else []
            
        for i in range(len(coords)):
            if (((yAvg[index] > yTop[i]) and (yAvg[index] < yBottom[i])) or 
                ((yTop[index] > yTop[i]) and (yTop[index] < yBottom[i]))  or
                ((yBottom[index] > yTop[i]) and (yBottom[index] < yBottom[i]))):
                continue
            else:
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
                    storeyCount+=1
        print('StoreyCount after running post processing module: ', storeyCount)
        print('StoreyHeights are: ', storeyHeights)
        return storeyCount, storeyHeights

    def plotBoxes(self, verticalPlane, boxes):
        for box in boxes:
            sX, sY, eX, eY = box
            cv2.rectangle(verticalPlane, (sX, sY), (eX, eY), (255, 255, 255), 4)
        
        fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(30, 800))
        ax1.imshow(verticalPlane, cmap="gray")
        ax1.set_xlabel("PLANE", fontsize=14)
        # plt.imshow(verticalPlane)
        # plt.tight_layout()
        plt.show()
        return verticalPlane

    def mapToPick(self, coordinates, padding = 5):
        pick = []
        for i in range(coordinates.shape[0]):
            startX, endX, startY, endY = self.calculateRange(coordinates[i], padding)
            pick.append([startX, startY, endX, endY])
        return pick

    def runCalculateTotalParamsModule(self):
        verticalPlaneCopy = np.copy(self.verticalPlane)
        FinalList = self.mapToAll4Coords(self.recordedCoordsOfEntireSeq)
        allMappedBoxes = []
        for i in range(len(FinalList)):
            ## MAIN LOOP ##
            coords1 = FinalList[i]
            
            ## Mapped to sx,sy,ex,ey Form
            pick1 = self.mapToPick(coords1, padding = 0)
            
            mappedBoxes = self.mapToVerticalPlane(pick1,self.height_info[i], verticalPlaneCopy)
            allMappedBoxes = allMappedBoxes + mappedBoxes
            
            # fig, ax1 = plt.subplots(nrows=1, ncols=1, constrained_layout=False, figsize=(25, 500))
            # ax1.imshow(verticalPlaneCopy, cmap="gray")
            # ax1.set_xlabel("PLANE", fontsize=14)
            # plt.show()
            
        ## NMS
        print("before NMS:", np.array(allMappedBoxes))
        self.plotBoxes(np.copy(self.verticalPlane), allMappedBoxes)
        print(np.array(allMappedBoxes).shape)

        finalBoxes = self.non_max_suppression_fast(np.array(allMappedBoxes))
        print("after NMS:", finalBoxes)
        print(finalBoxes.shape)
        finalMappedPlane = self.plotBoxes(np.copy(self.verticalPlane), finalBoxes)
        print("Total no. of windows in seq: " + str(finalBoxes.shape[0]))
        storeyCount, storeyHeights = self.calculateStoreys(finalBoxes)

        # finalMappedPlane = imutils.resize(finalMappedPlane, width=800)
        # cv2.imshow('Final IMAGE!!', finalMappedPlane)
        # #wait for a key to be pressed to exit
        # cv2.waitKey(0)
        # #close the window
        # cv2.destroyAllWindows()