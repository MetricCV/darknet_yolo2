import json
import numpy as np
import cv2
from shapely.geometry import box
import shapely
import os.path

def map_ann2yoloout(ann_path,relation_dict,image_path=None):
    # This file maps annotations of an image to its equivalent in pyyolo output format.
    #
    # Inputs:
    # -----
    # -ann_path: (str) is the path of the file with the annotations.
    # -image_path: (str) is the path of the image file, if none it assumes that it is the same path than ann_path but changes .txt by .jpg
    # -relationdict: (dict) is the mapping between the first column in ann [keys(relationdict)]
    #   file and the categories in detec_path (values(relationdict))
    # Output:
    # -------
    # -ann_list (list) with the same structure than pyyolo output
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
    ann_list=[]
    for i in ann_file:
        line=i.split()
        xhalf=int(np.floor(float(line[1])*w))
        yhalf=int(np.floor(float(line[2])*h))
        width=int(np.floor(float(line[3])*w))
        height=int(np.floor(float(line[4])*h))
        ann_list.append({'class':inv_relation_dict[int(line[0])],'left':int(xhalf-width*0.5),'right':int(xhalf+width*0.5),'top':int(yhalf+height*0.5),'bottom':int(yhalf-height*0.5),'prob':1})
    ann_file.close()
    return(ann_list)

def confusion_detection_ann_images(detect_path,ann_path,relationdict,image_path=None,Treshold=0.5):
    # This function is used to compute the confusion matrix associated to an annotated image (test),
    # and the detections made by a trained neural network in that image
    #
    # Inputs:
    # -------
    # -detect_path (str) is the path of the file with the detection (output of pyyolo)
    # -ann_path (str) is the path of the file with the annotations
    # -relationdict (dict) is the mapping between the first column in annotations files [values(relationdict)]
    #    file and the categories in detect_path (keys(relationdict))
    # Output:
    # --------
    # -act_pred_fp (numpy.array) matrix with where first axis represent the actual class and second axis represent the detected class 
    #   (false positives in the last row)
    # -act_pred_fn (numpy.array) matrix with where first axis represent the actual class and second axis represent the detected class 
    #   (false negatives in the last column)
    #inv_relationdict=dict(zip(list(relationdict.values()),list(relationdict.keys())))
    if os.path.isfile(detect_path)==True: 
        detect=json.load(open(detect_path,"r"))
    else:
        detect=[]  
    if os.path.isfile(ann_path)==False:        
        print("No annotations for image", ann_path.split(".")[0]+".jpg")
        ann=[]   
    else:
        ann=map_ann2yoloout(ann_path,relationdict, None)# mapping from annotation to pyyolo output
    act_pred_fp=np.zeros([len(relationdict.keys())+1,len(relationdict.keys())])# local matrix of false positives
    act_pred_fn=np.zeros([len(relationdict.keys()),len(relationdict.keys())+1])# local matrix of false negatives 
    pann=[]
    if len(ann)>0:
        if len(detect)>0:
            #creating IOU matrix
            IOU=np.zeros([len(ann),len(detect)])
            dann=[]
            for i in detect:
                dann.append(box(float(i["left"]),float(i["top"]),float(i["right"]),float(i["bottom"]))) # build a list of boxes associated to detections      
            for i in range(0,len(ann)):
                pann.append(box(float(ann[i]["left"]),float(ann[i]["top"]),float(ann[i]["right"]),float(ann[i]["bottom"])))
                for j in range(0,len(dann)):
                    inter_area=float(pann[i].intersection(dann[j]).area)
                    union_area=float(pann[i].union(dann[j]).area)
                    IOU[i,j]=inter_area/union_area
            #end IOU matrix 
            #false negative
            for i in range(0,len(ann)):
                argMAXIOU=np.argmax(IOU[i,:])
                MAXIOU=np.amax(IOU[i,:])
                num_class_ann=relationdict[ann[i]["class"]]
                num_class_detec=relationdict[detect[argMAXIOU]["class"]]
                if MAXIOU>=Treshold:
                    act_pred_fn[num_class_ann,num_class_detec]+=1# we go to the detected class
                else:
                    act_pred_fn[num_class_ann,len(relationdict.keys())]+=1# we go to the last entry which means that we didn't classify anything
            #false positive
            for i in range(0,len(detect)):
                argMAXIOU=np.argmax(IOU[:,i])
                MAXIOU=np.amax(IOU[:,i])
                num_class_ann=relationdict[ann[argMAXIOU]["class"]]
                num_class_detec=relationdict[detect[i]["class"]]
                if MAXIOU>=Treshold:
                    act_pred_fp[num_class_ann,num_class_detec]+=1# we go to the detected class
                else:
                    act_pred_fp[len(relationdict.keys()),num_class_detec]+=1# we go to the last entry which means that we didn't classify anything        
        else:
            for i in ann:    #what happend when we do not detect anything
                num_class_ann=relationdict[i["class"]]
                act_pred_fn[num_class_ann,len(relationdict.keys())]+=1 
    else:
        for i in detect:    #what happend when we do not detect anything
            num_class_detec=relationdict[i["class"]]
            act_pred_fp[len(relationdict.keys()),num_class_detec]+=1                  
    return(act_pred_fp,act_pred_fn)
  