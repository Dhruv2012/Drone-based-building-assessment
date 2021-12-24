import numpy as np 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_points = []

fig = plt.figure(figsize=(10,15))

name = "5"

path = './input_to_selectPts/'
investigate_outDir = './storedCoordinatesForDepthCalc/'
img=mpimg.imread(path +str(name)+'.jpg')
# dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

def onclick(event):
    ix, iy = event.xdata, event.ydata
    # print(ix, iy)
    image_points.append([ix, iy])

cid = fig.canvas.mpl_connect('button_press_event', onclick)

imgplot = plt.imshow(img)
plt.show()
np.savetxt(investigate_outDir+str(name)+'.out', image_points , delimiter=',')

print(end ="") 

image_points = np.loadtxt(investigate_outDir+str(name)+'.out', delimiter=',')

N = len(image_points)
image_points = np.array(image_points)
#print(image_points)
fig = plt.figure(figsize=(10,15))

img=mpimg.imread(path+str(name)+'.jpg')
imgplot = plt.imshow(img)

colors = np.random.rand(N)
area = (15 * np.ones(N))**2 

plt.scatter(image_points[:,0], image_points[:,1], c=colors, s=area)
#plt.plot(image_points[:,0], image_points[:,1])
plt.savefig(path+str(name)+'_dot.jpg')
plt.show()