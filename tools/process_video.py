import matplotlib
matplotlib.use('Agg')
import pyyolo
import numpy as np
import os
import sys
import glob
import cv2
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import subprocess


def annotate_image(im_data,yolo_class_name,yolo_class_color, output_dir="../results", detections=None, scale=1., sufix="1", color="blue", im_dpi=72,im_name="frame_",im_ext=".jpg"):
    im_file=os.path.join(output_dir, im_name+sufix+im_ext)
    im_shape=im_data.shape
    fig, ax = plt.subplots(1, 1, figsize=(im_shape[1]/im_dpi, im_shape[0]/im_dpi), frameon = False, dpi=im_dpi)
    #fig,ax = plt.subplots(figsize=(16,9), frameon=False)
    ax.imshow(im_data)
    class_of_intereset=list(yolo_class_name.keys())
    for detection in detections:
        if detection['class'] in class_of_intereset:
            r=int(detection['right'])/scale
            l=int(detection['left'])/scale
            t=int(detection['top'])/scale
            b=int(detection['bottom'])/scale
            name=yolo_class_name[detection['class']]
            color=yolo_class_color[detection['class']]
            proba=np.around(float(detection['prob']),decimals=2)
            rect = patches.Rectangle((l-4,t-3),r-l+8,b-t+4,linewidth=3,edgecolor=color,facecolor='none')      
            ax.add_patch(rect)
            label=ax.text(l-7, t-10, name+" Probability: "+str(proba), fontsize=14)
            label.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white'))
            #ax.annotate(detection['class'],(l-7,t-10),color='black', backgroundcolor='white',fontsize=14)

    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.savefig(im_file, dpi=im_dpi)
    plt.close()

if __name__ == "__main__":

    # yolo_class_color={
    #     'luber_texto':"blue",
    #     'luber_lubri':"blue",
    #     'luber_logo':"blue",
    #     'acdelco_logo':"red",
    #     'acdelco_baterias':"red",
    #     'tablero':"green"}

    # yolo_class_name={
    #     'luber_texto':"Luber",
    #     'luber_lubri':"Luber",
    #     'luber_logo':"Luber",
    #     'acdelco_logo':"ACDelco",
    #     'acdelco_baterias':"ACDelco",
    #     'tablero':"Tablero"}
    # darknet_path = '../'
    # data_file = 'cfg/futbol_mexico/yolo_metric_train.data'
    # cfg_file = 'cfg/futbol_mexico/yolo_metric.cfg'
    # weight_file = '/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_train_31000.weights'
    # video_file='/mnt/backup/NVR/futbol_mexico/Monterrey_vs_Tigres_C2017_small.mp4'
    # output_dir='../results/images_video'
    # output_video_file="../results/Monterrey_vs_Tigres_C2017_small_MetricCV.mp4"
    # output_video_fps=25
    # #output_file='../results/Monterrey_vs_Tigres_C2017_small_output_yolo.txt'
    # yolo_class_color={
    # "Head":"blue",
    # "Face":"red",
    # "Person":"green"
    # }
    # yolo_class_name={ 
    # 'Head':"Cabeza",
    # 'Face':"Cara",
    # 'Person':"Persona"
    # }
    # darknet_path = '../'
    # data_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/cfg/yolo_metric_train.data'
    # cfg_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/cfg/yolo_metric.cfg'
    # weight_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/yolo_metric_train_25000.weights'
    # video_file='/mnt/backup/NVR/vivo_coquimbo/cam8/20170928/01000001522000000.mp4'
    # output_dir='../results_arpon/annotations_vale_caro/images_video/vivo_coquimbo_cam8_20170928_01000001522000000'
    # output_video_file="../results_arpon/vivo_coquimbo_cam8_20170928_01000001522000000.mp4"
    # output_video_fps=30
    # output_file='../results_arpon/vivo_coquimbo_cam8_20170928_01000001522000000.txt'

    yolo_class_color={
    "person":"blue",
    "car":"red",
    "motorbike":"green",
    "bus":"yellow",
    "truck":"cyan"
    }
    yolo_class_name={ 
    "person":"Persona",
    "car":"Auto",
    "motorbike":"Moto",
    "bus":"Bus",
    "truck":"Camion"
    }
    darknet_path = '../'
    data_file = '/mnt/data/yolo/yolo.data'
    cfg_file = '/mnt/data/yolo/yolo.cfg'
    weight_file = '/mnt/data/yolo/yolo.weights'
    video_file='/home/sarpon/github/darknet_metric/results_arpon/tomas_francisco/20180109b/vid_23/vid_23.avi'
    output_dir='../results_arpon/tomas_francisco/20180109b/vid_23/images_video/'
    output_video_file="../results_arpon/tomas_francisco/20180109b/vid_23/annotated_vid_23.mp4"
    output_video_fps=10
    output_file='../results_arpon/20180109b_vid_23.txt'
    regular_name_of_frame='frame_'
    image_extension='.jpg'

    thresh = 0.25
    hier_thresh = 0.4

    # define initial values
    frame_id=0
    categories=set()
    storyofclass={}
    stop=0
    dataprev=0

    Create output folder
    if os.path.isdir(output_dir):
        for file in glob.iglob(os.path.join(output_dir, image_extension)):
            os.remove(file)
    else:
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    # Open video stream
    cap = cv2.VideoCapture(video_file) #opening the cam
    ret_val, img = cap.read()
    h, w, c = img.shape
    # ratio=np.min([540/float(h), 960/float(w)])

    # Load YOLO weight
    pyyolo.init(darknet_path, data_file, cfg_file, weight_file)#loading darknet in the memory

    time_start=datetime.now()
    while (cap.isOpened()):
        if frame_id % 100==0:
            print("Processing frame: ", frame_id)

        ret_val, img = cap.read()
        frame_id+=1
        if not ret_val:
            break

        #if frame_id % 5 != 0:
        #    continue

        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if ratio<1:
        #     img=cv2.resize(img,(0,0),fx=ratio,fy=ratio)
        #     h, w, c = img.shape
        img = img.transpose(2,0,1)
        data = img.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        outputs = pyyolo.detect(w, h, c, data, thresh, hier_thresh)
        if len(outputs)>0:
            for output in outputs:
                if (output["class"] in categories)==True:
                    storyofclass[output["class"]].append(frame_id)   
                else:
                    categories.add(output["class"])
                    storyofclass[output["class"]]=[frame_id]
        if len(outputs)>0:
            print("The frame_id=",frame_id," image contains detections")
        annotate_image(img_rgb,yolo_class_name,yolo_class_color, output_dir=output_dir, detections=outputs, scale=1, sufix="{0:06d}".format(frame_id),im_name=regular_name_of_frame,im_ext=image_extension)

    cap.release()
    time_end=datetime.now()
    print("Total execution time in minutes: ", (time_end-time_start).total_seconds()/60)

    json.dump(storyofclass,open(output_file,"w"))
    pyyolo.cleanup()
    regular_frame_names=regular_name_of_frame+'*'+image_extension
    # Create video from annotated images
    command="ffmpeg -y -r {0:d} -f image2 -pattern_type glob -i \"{1}\" -threads 8 -vcodec libx264 -crf 25 -pix_fmt yuv420p {2}".format(output_video_fps, os.path.join(output_dir,regular_frame_names), output_video_file)
    print(command)
    subprocess.call(command, shell=True)
