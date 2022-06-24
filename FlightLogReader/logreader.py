
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

#Reads logs for DJI drones. CSV files from both Air Data and Phantom can be read using the respective functions
#Units: time in ms, distances in meters and angles in degrees
def ReadAirDataLogs(log_path):
    
    logs = pd.read_csv(log_path, low_memory=False)
    logs.reset_index()
    
    #Readings corresponding to the following columns will be taken from the CSV files
    column_topics = ['time(millisecond)', 'latitude', 'longitude', 'height_above_takeoff(meters)',' xSpeed(m/s)',' ySpeed(m/s)', ' zSpeed(m/s)', ' pitch(degrees)', ' roll(degrees)',' compass_heading(degrees)', 'gimbal_heading(degrees)', 'gimbal_pitch(degrees)', 'distance(meters)']
    data_topics = ['time', 'latitude', 'longitude', 'height', 'xspeed', 'yspeed', 'zspeed', 'pitch', 'roll', 'yaw', 'gimbal.yaw','gimbal.pitch', 'distance(meters)']
    imu_readings = []
    for index, row in logs.iterrows():
        #Considering imu readings only for those instances when the video recording was on
        if row['isVideo'] == 1:
            readings = {}
            for column_topic, data_topic in zip(column_topics,data_topics):
                readings[data_topic]=row[column_topic]
            imu_readings.append(readings)
    
    return imu_readings

def ReadPhantomLogs(log_path):
    logs = pd.read_csv(log_path, low_memory=False, header=1)
    logs.reset_index()
    
    #Readings corresponding to the following topics will be taken from the CSV files
    column_topics = [' OSD.flyTime [s]', ' OSD.latitude', ' OSD.longitude', ' OSD.height [ft]', ' OSD.xSpeed [MPH]', ' OSD.ySpeed [MPH]', ' OSD.zSpeed [MPH]', ' OSD.pitch', ' OSD.roll', ' OSD.yaw [360]', ' GIMBAL.pitch', ' GIMBAL.yaw [360]', 'distance(meters)']
    data_topics = ['time', 'latitude', 'longitude', 'height', 'xspeed', 'yspeed', 'zspeed', 'pitch', 'roll', 'yaw', 'gimbal.yaw','gimbal.pitch','distance(meters)']

    imu_readings = []
    for index, row in logs.iterrows():
        #Considering imu readings only for those instances when the video recording was on
        if row[' CAMERA.isVideo'] == True:
            readings = {}
            for column_topic, data_topic in zip(column_topics,data_topics):
                if 'height' in column_topic:
                    #The unit of height in phantom logs in ft, converting it to meters
                    readings[data_topic]  = row[column_topic] * 0.3048
                elif 'Speed' in column_topic:
                    #The unit of speed in phatom logs in miles/hour, converting it to meters/second
                    readings[data_topic] = row[column_topic] * 0.44704
                elif 'Time' in column_topic:
                    readings[data_topic] = row[column_topic]*1000
                else:
                    readings[data_topic]=row[column_topic]
            imu_readings.append(readings)

    return imu_readings

def getFrame(video, sec, image_directory, count, time_interval):
    video.set(cv2.CAP_PROP_POS_MSEC,sec*time_interval)
    hasFrames,image = video.read()
    if hasFrames:
        image_path = image_directory+'/'+str(count)+".jpg"
        # local_image_path = 'images/' + str(count)+".jpg"
        image_names.append(image_path)
        cv2.imwrite(image_path, image)     # save frame as JPG file
    return hasFrames

def SampleVideo(video_path, time_interval=100):
    global image_names 
    image_names = []
    #Input - Path of the video that needs to be sampled, time_interval(ms): decides the number of images required
    video = cv2.VideoCapture(video_path)
    directory_path = os.path.dirname(video_path)
    image_directory = os.path.join(directory_path, 'images')
    
    #Creating a directory within the parent directory, to store the images
    if not os.path.exists(image_directory):
        try:
            original_umask = os.umask(0)
            os.makedirs(image_directory, 0o0777)
        finally:
            os.umask(original_umask)
    sec = 0
    frameRate = 1
    count = 1
    success= getFrame(video, sec,image_directory,count,time_interval)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec,2)
        success= getFrame(video, sec,image_directory,count,time_interval)

    return image_names, directory_path

def GetTimeOfImage(image_names_list,logs, time_interval):
    
    time_synced_images = []
    skip = int(time_interval/100)
    
    #Skipping readings as per time_interval, flight logs have a fixed_time interval of 100ms
    print(skip)
    logs_new = logs[::skip]
    print(len(logs))
    print(len(logs_new))
    total_images = len(image_names_list)
    total_readings = len(logs_new)

    #Matching the number of readings and images, dropping the last 'diff' readings
    diff = -int(total_readings - total_images)
    print(diff)
    if diff!=0:
        logs_new = logs_new[:diff]
    # for i, j in zip(image_names_list,logs_new):
    #     #Converting the time to seconds
    #     time_synced_images.append([float(j['time']/1000), i])
    
    # return time_synced_images
    
    for i,j in zip(image_names_list, logs_new):
        j['image_name'] = i
    
    return logs_new

def WriteTextFile(time_synced_images, directory_path):
    os.chdir(directory_path)
    with open('images.txt', 'w') as file:
        for item in time_synced_images:
            file.write(" ".join(map(str,item)))
            file.write("\n")



if __name__ == "__main__":

    log_path = '/home/kushagra/Downloads/Jun-23rd-2022-03-58PM-Flight-Airdata.csv'
    airdatalogs = ReadAirDataLogs(log_path)

    # print(len(airdatalogs))
    video_path = '/home/kushagra/IIIT-H/ObjectDetectionDatasets/DJI_0641/DJI_0641.MP4'
    time_interval = 100
    image_names_list, diretory_path = SampleVideo(video_path, time_interval)
    print(diretory_path)

    image_synced_logs = GetTimeOfImage(image_names_list, airdatalogs, time_interval)

    # print(time_synced_images)
    # WriteTextFile(time_synced_images, diretory_path)
    data = pd.DataFrame(image_synced_logs)
    data.to_csv(diretory_path+'/data.csv')