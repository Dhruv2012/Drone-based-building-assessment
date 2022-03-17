"""
Downsamples and saves a video.
Project: Building Inspection using Drones - IIITH
"""

import sys
import argparse
import cv2

print(cv2.__version__)

def downsampleVideo(pathIn, pathOut, freq=1):
	count = 0
	writeCount = 0
	freq = int(freq)
	capture_obj = cv2.VideoCapture(pathIn)
	ret, frame = capture_obj.read()
	while ret:
		if not ret:
			continue
		if ((count+1)%freq == 0):
			cv2.imwrite(pathOut + "//%#05d.jpg"%(writeCount+1), frame)
			writeCount += 1
		ret, frame = capture_obj.read()
		count += 1
	print("Downsampled {} into {} frames at {}".format(pathIn, count, pathOut))

if __name__ == "__main__":
	arg = argparse.ArgumentParser()
	arg.add_argument("--pathIn", help="path to video")
	arg.add_argument("--pathOut", help="path to images")
	arg.add_argument("--freq", help="Downsample frequency")
	args = arg.parse_args()
	print(args)
	downsampleVideo(args.pathIn, args.pathOut, args.freq)