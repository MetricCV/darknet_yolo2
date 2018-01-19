import json
from cv2 
import numpy as np
from shapely.geometry import box
import shapely
def map_ann2yoloout(ann_path=None,image_path=None,relation_dict):
'''
Inputs:
-----
ann_path: (str) is the path of the file with the annotations.

image_path: (str) is the path of the image file, if none it assumes that it is the same path than ann_path but changes .txt by .jpg

relationdict: (dict) is the mapping between the first column in ann [keys(relationdict)] file and the categories in detec_path (values(relationdict))

Output:
-------
(dict) with the same structure than the darknetout
''' 
    inv_relation_dict=dict(zip(list(relation_dict.values()),list(relation_dict.keys())))
    if image_path=None:
        image_path=ann_path.split(".")[0]+"jpg"
    img=cv2.imread(image_path)
    if img!=None:
        h,w,c=img.shape
    else:
	print("Image is not in Folder")
	return(None)
    ann_file=open(ann_path,"r")
    ann_dict=[]
    for i in ann_file:
        line=i.split()
        left=np.floor(int(line[1])*w)
        top=np.floor(int(line[2])*h)
        right=np.floor(int(line[3])*w)
        bottom=np.floor(int(line[4])*h)
        ann_dict.append({'class':inv_relation_dict[int(line[0])],'left'=left,'right'=right,'top'=top,'bottom'=bottom,'prob'=1})
    return(ann_dict)
def confusion_detection_ann_images(detec_path,ann_path,image_path=None,relationdict,Treshold=0.5):
'''
Inputs:
-------
detect_path (str) is the path of the file with the detection (output of pyyolo)

ann_path (str) is the path of the file with the annotations

relationdict (dict) is the mapping between the first column in ann [values(relationdict)] file and the categories in detec_path (keys(relationdict))

Output:
--------
act_pred (numpy.array) matrix with where first axis represent the actual class and second axis represent the detected class
'''
    inv_relationdict=dict(zip(list(relationdict.values()),list(relation_dict.keys())))
    detec=json.load(open(detect_path,"r"))
    ann=map_ann2yoloout(ann_path,,relationdict)
    act_pred=np.zeros([len(relationdict.keys())+1,len(relationdict.keys())+1])
    pann=[]
    dann=[]
    if len(detect)>0:    
	    for i in ann:
	        pann.append(box(float(i["left"]),float(i["top"]),float(i["right"]),float(i["bottom"])))
	        IOU=np.zeros([len(ann),len(detect)])
	        for j in detect:
	            dann.append(box(float(j["left"]),float(j["top"]),float(j["right"]),float(j["bottom"])))
	        for j in range(0,len(pann)):
	            for k in range(0,len(dann)):
	                IOU[j,k]=(pann[j].intersection(dann[k]))/(pann[j].union(dann[k]))
	        argMAXIOU=np.argmax(IOU)
	        MAXIOU=np.amax(IOU)
	        for j in range(0,len(pann)):
	            num_class_ann=relationdict[ann[j]["class"]]
	            num_class_detec=relationdict[detec[argMAXIOU[j]]["class"]]
	            if MAXIOU[j]>=Treshold:
	                act_pred[num_class_ann,num_class_detec]+=1# we go to the detected class
	            else
	            	act_pred[num_class_ann,len(relationdict.keys())]+=1# we go to the last entri which means that we didn't classify anything        
    else:
    	for i in ann:    #what happend when we do not detect anything
    		num_class_ann=relationdict[i["class"]]
    		act_pred[num_class_ann,len(relationdict.keys())]+=1	
    return(act_pred)