import glob
import cv2
import numpy as np
import os
size = 10
# generating the kernel
filters=[]
kernel_motion_blur0 = np.zeros((size, size))
kernel_motion_blur0[:,int((size-1)/2)] = np.ones(size)
kernel_motion_blur0 = kernel_motion_blur0 / size
filters.append(kernel_motion_blur0)
kernel_motion_blur1 = np.zeros((size, size))
kernel_motion_blur1[int((size-1)/2),:] = np.ones(size)
kernel_motion_blur1 = kernel_motion_blur1 / size
filters.append(kernel_motion_blur1)

size1 = 8
kernel_motion_blur2 = np.eye(size)
kernel_motion_blur2 = kernel_motion_blur2 / size
filters.append(kernel_motion_blur2)
kernel_motion_blur3 = np.eye(size)
kernel_motion_blur3 = np.fliplr(kernel_motion_blur3)
kernel_motion_blur3 = kernel_motion_blur3 / size
filters.append(kernel_motion_blur3)

path="yolo"
files=glob.glob(path+"/*.jpg")
for count,i in enumerate(files):
	print(count)
	img=cv2.imread(i)
	output=cv2.filter2D(img,-1,filters[int(np.random.randint(4))])
	os.remove(i)
	cv2.imwrite(i.split(".")[0]+"_blur.jpg",output)
	nameann_in=i.replace(".jpg",".txt")
	nameann_out=i.replace(".jpg","_blur.txt")
	os.rename(nameann_in,nameann_out)
	
