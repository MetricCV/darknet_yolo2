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

def annotate_image(im_data, detections=None, scale=1., save_name="1",color="blue", im_dpi=72):
    im_shape=im_data.shape
    fig, ax = plt.subplots(1, 1, figsize=(im_shape[1]/im_dpi, im_shape[0]/im_dpi), frameon = False, dpi=im_dpi)
    #fig,ax = plt.subplots(figsize=(16,9), frameon=False)
    ax.imshow(im_data)

    for detection in detections:
        r=int(detection['right'])/scale
        l=int(detection['left'])/scale
        t=int(detection['top'])/scale
        b=int(detection['bottom'])/scale
        name=yolo_class_name[detection['class']]
        color=yolo_class_color[detection['class']]

        rect = patches.Rectangle((l-4,t-3),r-l+8,b-t+4,linewidth=3,edgecolor=color,facecolor='none')      
        ax.add_patch(rect)
        label=ax.text(l-7, t-10, name, fontsize=14)
        label.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
        #ax.annotate(detection['class'],(l-7,t-10),color='black', backgroundcolor='white',fontsize=14)

    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(save_name, dpi=im_dpi) 
    plt.close()

if __name__ == "__main__":
    '''
    This function takes a text file in image_path with the path of images to annotate it

    Return:
    Anotated imagen in jpg format and text file where: first column is the corresponding class(int) second and third column are "x" and "y" of the left top corner of the box, fourth and fifth columns are "x" and "y" of the right bottom corner of the box. This coordinates are absolute with respect to the original imagessize and integers
    '''
    images_path= '../cfg/futbol_mexico/test.txt' 
    darknet_path = '../'
    data_file = 'cfg/futbol_mexico/yolo_metric.data'
    cfg_file = 'cfg/futbol_mexico/yolo_metric.cfg'
    weight_file = '/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_31000.weights'
    output_file='../results/Monterrey_vs_Tigres_C2017_small_output_yolo.txt'
	
    thresh = 0.24
    hier_thresh = 0.5

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
        print(i)
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
        print(outputs)
        input("pausa")
        if len(outputs)>0:
            if (outputs[0]["class"] in categories)==True:
                storyofclass[outputs[0]["class"]].append(frame_id)   
            else:
                categories.add(outputs[0]["class"])
                storyofclass[outputs[0]["class"]]=[frame_id]
            print("The frame_id=",frame_id," image contains detections")
            for j in range(0,len(outputs)):	
                outputs[j]["left"]=np.floor(float(outputs[j]["left"])/ratio)
                outputs[j]["right"]=np.floor(float(outputs[j]["right"])/ratio)
                outputs[j]["top"]=np.floor(float(outputs[j]["top"])/ratio)
                outputs[j]["bottom"]=np.floor(float(outputs[j]["bottom"])/ratio)
        name=i.split(".")[0]+"detection.jpg"
        annotate_image(img_rgb, detections=outputs, scale=ratio, save_name=name,im_dpi=72)
        name=i.split(".")[0]+"detection.txt"
        json.dump(outputs,open(name,"w")) 
    time_end=datetime.now()
    print("Total execution time in minutes: ", (time_end-time_start).total_seconds()/60)
    pyyolo.cleanup()
