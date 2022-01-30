import sys
import argparse
import cv2
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


print(cv2.__version__)

'''
Colmap stores the transformation from world coordinates to the local camera 
coordinates, i.e., from a 3D coordinate system to a 3D coordinate system. 
If you take the rotation R and the translation t that are stored in the images.txt
file, then concatenating them into [R|t] will give you this transformation.
'''

def getR_from_q(q):
	# w, x, y, z
	# https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
	return np.array([
		[-1 + 2 * q[0] ** 2 + 2 * q[1] ** 2,
		 2 * q[1] * q[2] - 2 * q[0] * q[3],
		 2 * q[3] * q[1] + 2 * q[0] * q[2]],

		[2 * q[1] * q[2] + 2 * q[0] * q[3],
		 -1 + 2 * q[0] ** 2 + 2 * q[2] ** 2,
		 2 * q[2] * q[3] - 2 * q[0] * q[1]],

		[2 * q[3] * q[1] - 2 * q[0] * q[2],
		 2 * q[2] * q[3] + 2 * q[0] * q[1],
		 -1 + 2 * q[0] ** 2 + 2 * q[3] ** 2]])


def ReadCameraOrientation(pathIn):
	with open(pathIn) as f:
		lines = f.readlines()
	print(len(lines))

	line_count = 0 # Every odd line needs
	Rs = []
	ts = []
	only_transformation_lines = []

	for index, line in enumerate(lines):
		line = line.strip()

		if not line.startswith('#'):
			line_count = line_count + 1

			if line_count % 2 == 1:
				elements = line.split(" ")
				only_transformation_lines.append(elements)

	# print(only_transformation_lines)
	only_transformation_lines.sort(key=lambda x: int(x[0]))

	for line in only_transformation_lines:
		q = []
		for i in range(1,5):
			q.append(float(line[i]))
		t = []
		for j in range(5,8):
			t.append(float(line[j]))
		print(q, t)

		R = getR_from_q(q)
		t = (np.array(t)).T
		# print(R)
		# print(t)
		Rs.append(R)
		ts.append(t)
	# print()

	return Rs, ts


def getH_from_R_t(R, t):
	# assuming R, t are numpy array
    h = np.column_stack((R.T, -R.T@t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    assert h.shape == (4,4)
    return h


		
def PlotOdometry(Rs, ts):
	ax = plt.axes(projection='3d')

	oldHomogeneousMatrix = np.identity(4)
	oldTranslationMatrix = np.array([[0,0,0,1]]) # origin
	oldTranslationMatrix = oldTranslationMatrix.T

	for R,t in zip(Rs, ts):
		# print(R)
		# print(t)
		H = getH_from_R_t(R, t)
		# print(H)
		oldHomogeneousMatrix = H # oldHomogeneousMatrix @  H
		pose = oldHomogeneousMatrix @ oldTranslationMatrix
		# print("pose : ", pose, "shape", pose.shape)
		ax.scatter(pose[0][0], pose[1][0], pose[2][0], cmap='green')
	plt.show()




if __name__ == "__main__":
	arg = argparse.ArgumentParser()
	arg.add_argument("--pathIn", help="path to Images.txt")
	args = arg.parse_args()
	print(args)
	Rs, ts = ReadCameraOrientation(args.pathIn)
	PlotOdometry(Rs, ts)
