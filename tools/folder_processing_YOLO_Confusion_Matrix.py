import matplotlib
matplotlib.use('Agg')
import pyyolo
import numpy as np
import sys
import cv2
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import confusion_matrix as conf_mtrx
from tempfile import TemporaryFile
import os.path
yolo_class_color={
    'luber_texto':"blue",
    'luber_lubri':"blue",
    'luber_logo':"blue",
    'acdelco_logo':"red",
    'acdelco_baterias':"red"}


yolo_class_name={
    'luber_texto':"Luber",
    'luber_lubri':"Luber",
    'luber_logo':"Luber",
    'acdelco_logo':"ACDelco",
    'acdelco_baterias':"ACDelco"}

def annotate_image(im_data, relation_dict, detections=None, scale=1., save_name="1",color="blue", im_dpi=72,annotation=None):
    if len(detections)>0:
        im_shape=im_data.shape
        fig, ax = plt.subplots(1, 1, figsize=(im_shape[1]/im_dpi, im_shape[0]/im_dpi), frameon = False, dpi=im_dpi)
        #fig,ax = plt.subplots(figsize=(16,9), frameon=False)
        ax.imshow(im_data)

        for detection in detections:
            r=int(detection['right'])/scale
            l=int(detection['left'])/scale
            t=int(detection['top'])/scale
            b=int(detection['bottom'])/scale
            name=yolo_class_name[detection['class']]+" "
            color=yolo_class_color[detection['class']]
            proba=np.around(float(detection['prob']),decimals=2)
            rect = patches.Rectangle((l-4,t-3),r-l+8,b-t+4,linewidth=3,edgecolor=color,facecolor='none')      
            ax.add_patch(rect)
            label=ax.text(l-7, t-10, name+"Probability: "+str(proba), fontsize=14)
            label.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
        if os.path.isfile(annotation)==True:    
            annotation_list=conf_mtrx.map_ann2yoloout(annotation,relation_dict,image_path=None)
            for i in annotation_list:
                #ax.annotate(detection['class'],(l-7,t-10),color='black', backgroundcolor='white',fontsize=14)
                r=int(i['right'])
                l=int(i['left'])
                t=int(i['top'])
                b=int(i['bottom'])
                name=yolo_class_name[i['class']]+" "
                color=yolo_class_color[i['class']]
                proba=float(i['prob'])
                rect = patches.Rectangle((l-4,t-3),r-l+8,b-t+4,linewidth=3,edgecolor=color,facecolor='none',linestyle="dashed")      
                ax.add_patch(rect)
                label=ax.text(l-7, t-10, name+"Annotation", fontsize=14)
                label.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))    
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.savefig(save_name, dpi=im_dpi) 
        plt.close()

if __name__ == "__main__":
    # This function takes a text file in image_path with the path of images to annotate it,
    # return a confusion matrix and an .npy where the matrix is saved.
    #
    # Input:
    # ------
    # -images_path (str) path to file wich indicates the test images.
    # -darknet_path (str) path to darknet folder.
    # -data_file (str) path to data file.
    # -cfg_file (str) path to condiguration file.
    # -weight_file (str) path to weight file.
    # -output_path (str) path where all the results will be saved.
    # -diff_clases_linknum (dict) mapp from string categories to numeric categories.
    # -hier_thres (float)
    # -thres (float) this parameters define the minimun probability that will be accepted as a detection.
    # -IOUTHREs (float) this parameter define the minimum IOU that will be accepted to assing to a class.
    # Output:
    # -------
    # -Anotated imagen in jpg format and text file where: first column is the corresponding class(int) 
    #   second and third column are "x" and "y" of the left top corner of the box, fourth and fifth 
    #   columns are "x" and "y" of the right bottom corner of the box. This coordinates are absolute with 
    #   respect to the original imagessize and integers
    # -sumaconfmatrix_3d_fp (numpy array) 
    # -outputfile (.npy) saving sumaconfmatrix_3d as numpy array

    images_path= '../cfg/futbol_mexico/test.txt' 
    darknet_path = '../'
    data_file = 'cfg/futbol_mexico/yolo_metric.data'
    cfg_file = 'cfg/futbol_mexico/yolo_metric.cfg'
    weight_file = '/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_31000.weights'
    output_path='../results_arpon/'
    diff_clases_linknum={
    'luber_texto':0,
    'luber_lubri':1,
    'acdelco_logo':2,
    'luber_logo':3,
    'acdelco_baterias':4
    'tablero':5}
    hier_thresh =0.5
    thresh = 0.5
    IOUTHRES=0.5

    # define initial values
    frame_id=0
    categories=set()
    storyofclass={}
    stop=0
    dataprev=0

    f=open(images_path,"r")
    im_path=f.readline().strip()
    img=cv2.imread(im_path)
    h, w, c = img.shape
    ratio=np.min([540/float(h), 960/float(w)])

    # Load YOLO weight
    pyyolo.init(darknet_path, data_file, cfg_file, weight_file)#loading darknet in the memory

    time_start=datetime.now()
    for i in f:
        i=i.strip()
        frame_id+=1
        if frame_id % 100==0:
            print("Processing frame: ", frame_id)
        img=cv2.imread(i)
        if type(img)==None:
            continue
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if ratio<1:
            img=cv2.resize(img,(0,0),fx=ratio,fy=ratio)
            h, w, c = img.shape
        img = img.transpose(2,0,1)
        data = img.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
        if len(outputs)>0:
            if (outputs[0]["class"] in categories)==True:
                storyofclass[outputs[0]["class"]].append(frame_id)   
            else:
                categories.add(outputs[0]["class"])
                storyofclass[outputs[0]["class"]]=[frame_id]
            print(i)
            print("The frame_id=",frame_id," image contains detections")
        name=i.split(".")[0]+"detection.jpg"
        aname=i.split(".")[0]+".txt"
        annotate_image(img_rgb, relation_dict=diff_clases_linknum, detections=outputs, scale=ratio, save_name=name,im_dpi=72,annotation=aname)
        if len(outputs)>0:
            for j in range(0,len(outputs)):   
                outputs[j]["left"]=np.floor(float(outputs[j]["left"])/ratio)
                outputs[j]["right"]=np.floor(float(outputs[j]["right"])/ratio)
                outputs[j]["top"]=np.floor(float(outputs[j]["top"])/ratio)
                outputs[j]["bottom"]=np.floor(float(outputs[j]["bottom"])/ratio)
        name=i.split(".")[0]+"detection.txt"
        localfile=open(name,"w")
        json.dump(outputs,localfile)
        localfile.close() 
    time_end=datetime.now()
    print("Total execution time in minutes: ", (time_end-time_start).total_seconds()/60)
    pyyolo.cleanup()
    f.seek(0)
    #building confusion matrix
    num_lines = sum(1 for line in f)
    confmatrix_3d_fp=np.zeros((len(diff_clases_linknum.keys())+1,num_lines,len(diff_clases_linknum.keys())))
    count=0
    f.seek(0)
    for i in f:
        nameann=i.split(".")[0]+".txt"
        namedetec=i.split(".")[0]+"detection.txt"
        [confmatrix_3d_fp[:,count,:],trash]=conf_mtrx.confusion_detection_ann_images(namedetec,nameann,diff_clases_linknum,None,IOUTHRES)
        count+=1
    sumaconfmatrix_3d_fp=np.zeros((len(diff_clases_linknum.keys())+1,len(diff_clases_linknum.keys())))
    for i in range(0,num_lines):
        sumaconfmatrix_3d_fp+=confmatrix_3d_fp[:,i,:]
    f.close()
    savename=str(output_path+"/ConfusionMatrix_Detthres_"+str(thresh)+"_IOUthres_"+str(IOUTHRES))
    print("======")
    print("(",thresh,",",IOUTHRES,")")
    print(sumaconfmatrix_3d_fp)
    mAP=0
    for i in range(0,np.shape(sumaconfmatrix_3d_fp)[1]-1):
        mAP+=((sumaconfmatrix_3d_fp[i,i])/(sum(sumaconfmatrix_3d_fp[:,i])))
    mAP=mAP/(np.shape(sumaconfmatrix_3d_fp)[0]-1)    
    print("======")
    np.save(savename,sumaconfmatrix_3d_fp,allow_pickle=False)
