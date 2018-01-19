import json
import numpy as np
import cv2
from shapely.geometry import box
import shapely
def map_ann2yoloout(ann_path,relation_dict,image_path=None):
    # Inputs:
    # -----
    # ann_path: (str) is the path of the file with the annotations.

    # image_path: (str) is the path of the image file, if none it assumes that it is the same path than ann_path but changes .txt by .jpg

    # relationdict: (dict) is the mapping between the first column in ann [keys(relationdict)] file and the categories in detec_path (values(relationdict))

    # Output:
    # -------
    # (dict) with the same structure than the darknetout
    inv_relation_dict=dict(zip(list(relation_dict.values()),list(relation_dict.keys())))
    if image_path==None:
        image_path=ann_path.split(".")[0]+".jpg"
    img=cv2.imread(image_path)
    if img is not None:
        h,w,c=img.shape
    else:
        print("Image is not in Folder")
        return(None)
    ann_file=open(ann_path,"r")
    ann_dict=[]
    for i in ann_file:
        line=i.split()
        xhalf=int(np.floor(float(line[1])*w))
        yhalf=int(np.floor(float(line[2])*h))
        width=int(np.floor(float(line[3])*w))
        height=int(np.floor(float(line[4])*h))
        ann_dict.append({'class':inv_relation_dict[int(line[0])],'left':(xhalf-width*0.5),'right':(xhalf+width*0.5),'top':(yhalf-width*0.5),'bottom':(yhalf+width*0.5),'prob':1})
    return(ann_dict)
def confusion_detection_ann_images(detect_path,ann_path,relationdict,image_path=None,Treshold=0.5):
    # Inputs:
    # -------
    # detect_path (str) is the path of the file with the detection (output of pyyolo)

    # ann_path (str) is the path of the file with the annotations

    # relationdict (dict) is the mapping between the first column in annotations files [values(relationdict)] file and the categories in detect_path (keys(relationdict))

    # Output:
    # --------
    # act_pred (numpy.array) matrix with where first axis represent the actual class and second axis represent the detected class
    inv_relationdict=dict(zip(list(relationdict.values()),list(relationdict.keys())))
    try: 
        detect=json.load(open(detect_path,"r"))
    except:
        detect=[]    
    ann=map_ann2yoloout(ann_path,relationdict, None)# mapping from annotation to pyyolo output
    act_pred=np.zeros([len(relationdict.keys()),len(relationdict.keys())+1])# local matrix of 
    pann=[]
    if len(detect)>0:
        dann=[]
        for j in detect:
            dann.append(box(float(j["left"]),float(j["top"]),float(j["right"]),float(j["bottom"]))) # build a list of boxes associated to detections      
        anncounter=0
        for i in ann:
            pannbox=box(float(i["left"]),float(i["top"]),float(i["right"]),float(i["bottom"]))
            IOU=np.zeros([len(ann),len(detect)])
            for k in range(0,len(dann)):
                inter_area=float(pannbox.intersection(dann[k]).area)
                union_area=float(pannbox.union(dann[k]).area)
                IOU[anncounter,k]=inter_area/union_area
            argMAXIOU=np.argmax(IOU,axis=1)
            MAXIOU=np.amax(IOU,axis=1)
            num_class_ann=relationdict[ann[anncounter]["class"]]
            num_class_detec=relationdict[detect[argMAXIOU[anncounter]]["class"]]
            if MAXIOU[anncounter]>=Treshold:
                act_pred[num_class_ann,num_class_detec]+=1# we go to the detected class
            else:
                act_pred[num_class_ann,len(relationdict.keys())]+=1# we go to the last entri which means that we didn't classify anything
            anncounter+=1                
    else:
        for i in ann:    #what happend when we do not detect anything
            num_class_ann=relationdict[i["class"]]
            act_pred[num_class_ann,len(relationdict.keys())]+=1      
    return(act_pred)
  