import pyyolo
import numpy as np
import sys
import cv2
from datetime import datetime
import json

darknet_path = '../'
data_file = 'cfg/futbol_mexico/yolo_metric.data'
cfg_file = 'cfg/futbol_mexico/yolo_metric.cfg'
weight_file = '/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_31000.weights'
video_file='/mnt/backup/NVR/futbol_mexico/Monterrey_vs_Tigres_C2017_small.mp4' 
output_file='../results/Monterrey_vs_Tigres_C2017_small_output_yolo.txt'

thresh = 0.24
hier_thresh = 0.5
cap = cv2.VideoCapture(video_file)#opening the cam
ret_val, img = cap.read()
h, w, c = img.shape
ratio=np.min([540/float(h), 960/float(w)])

pyyolo.init(darknet_path, data_file, cfg_file, weight_file)#loading darknet in the memory

# define initial values
fpcount=0
categories=set()
storyofclass={}
stop=0
dataprev=0

time_start=datetime.now()
while (cap.isOpened()):
	fpcount+=1
	ret_val, img = cap.read()
	if not ret_val:
		break
	if ratio<1:
		img=cv2.resize(img,(0,0),fx=ratio,fy=ratio)
		h, w, c = img.shape
	img = img.transpose(2,0,1)
	data = img.ravel()/255.0
	data = np.ascontiguousarray(data, dtype=np.float32)
	outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
	if len(outputs)>0:
		if (outputs[0]["class"] in categories)==True:
			storyofclass[outputs[0]["class"]].append(fpcount)	
		else:
			categories.add(outputs[0]["class"])
			storyofclass[outputs[0]["class"]]=[fpcount]
	if fpcount % 100==0:
		print(fpcount)

cap.release()
time_end=datetime.now()
print("Total execution time in minutes: ", (time_end-time_start).total_seconds()/60)

json.dump(storyofclass,open(output_file,"w"))
pyyolo.cleanup()