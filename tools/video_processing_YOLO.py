import pyyolo
import numpy as np
import sys
import cv2
from datetime import datetime
import json
darknet_path = '../'
datacfg = darknet_path +'cfg/futbol_mexico/yolo_metric.data'
cfgfile = darknet_path +'cfg/futbol_mexico/yolo_metric.cfg'
weightfile = '/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_31000.weights'
videoname='/mnt/data/dataset/futbol_mexico/Monterrey_vs_Tigres_C2017_small.mp4'#assing a video 
outputpath=darknet_path +'results_arpon/Monterrey_vs_Tigres_C2017_small_output_yolo.txt'
thresh = 0.24
hier_thresh = 0.5
cam = cv2.VideoCapture(videoname)#opening the cam
ret_val, img = cam.read()
h, w, c = img.shape
ratio=np.min([540/float(h), 960/float(w)])
pyyolo.init(darknet_path, datacfg, cfgfile, weightfile)#loading darknet in the memory
# camera
fpcount=0
categories=set()
storyofclass={}
stop=0
dataprev=0
start=datetime.now()
while (cam.isOpened()):
	fpcount+=1
	ret_val, img = cam.read()
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
	if fpcount % 1000==0:
		print(fpcount)
json.dump(storyofclass,open(outputpath,"w"))
pyyolo.cleanup()
