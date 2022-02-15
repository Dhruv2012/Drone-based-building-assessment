import cv2
vidcap = cv2.VideoCapture(r'F:\IIIT-H Work\win_det_heatmaps\datasets\IIIT-H Dataset\DJI_Dataset\top-view\All\DJI_0196.MP4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("DJI_0196_"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)