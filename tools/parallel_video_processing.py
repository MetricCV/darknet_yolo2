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
import multiprocessing as mp
import time


def annotate_image(im_data,yolo_class_color,yolo_classes_name,output_dir="../results", detections=None, scale=1., sufix="1", color="blue", im_dpi=72):
    im_file=os.path.join(output_dir, 'futbol_mexico_img'+sufix+'.jpg')
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

def capture(number_of_detectorsl,eventol,the_queuel,video_filel):
    # Inputs:
    # =======
    # vid_fil (str) video file path.
    # the_q (multiprocessing.Queue) the queue where the processed data waits.
    # Returns:
    # ========
    # [w, h, c, data] (list) add to the_q the list with the resized image(data) and the width (w), height(h) and number of channels(c)
    cap = cv2.VideoCapture(video_filel) #opening the cam
    frame_id=0
    ret_val, img = cap.read()
    h, w, c = img.shape
    # ratio=np.min([540/float(h), 960/float(w)]) 
    while (cap.isOpened()):
        if frame_id>200:
            break
        if frame_id % 100==0:
            print("Processing frame: ", frame_id)
        ret_val, img = cap.read()
        frame_id+=1
        if not ret_val:
            break
        img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if ratio<1:
            # img=cv2.resize(img,(0,0),fx=ratio,fy=ratio)
            # h, w, c = img.shape
        img = img.transpose(2,0,1)
        data = img.ravel()/255.0
        data = np.ascontiguousarray(data, dtype=np.float32)
        the_queuel.put([w, h, c, data,frame_id,img_rgb])
        # print(the_queuel.qsize())    
    cap.release()
    # for i in range(number_of_detectorsl):
    #     the_queuel.put(None)
    eventol.set()

def detect_image(integer,lokl,eventol,evento2l,the_queuel,the_final_queuel,threshl,hier_threshl,darknet_pathl, data_filel, cfg_filel, weight_filel):
    # Inputs:
    # =======
    # the_queuel(multiprocessing.Queue) the queue where the images waits to be processed.
    # hier_thresl (float)
    # thresl (float) this parameters define the minimun probability that will be accepted as a detection.
    # darknet_pathl (str) path to darknet folder.
    # data_filel (str) path to data file.
    # cfg_filel (str) path to condiguration file.
    # weight_filel (str) path to weight file.
    # Returns:
    # ========
    # outputs (list) every entry list is a dictionary with keys=['right','left','top','bottom','class','prob'] added to a queue
    # Load YOLO weight
    pyyolo.init(darknet_pathl, data_filel, cfg_filel, weight_filel)#loading darknet in the memory
    #while True:
    while the_queuel.qsize()>0 or eventol.is_set()==False:
        from_queue = the_queuel.get()  #note that every item in queuel is a list with the following form [w, h, c, data,frame_id,img_rgb]
        outpyyolo=pyyolo.detect(from_queue[0], from_queue[1], from_queue[2], from_queue[3], threshl, hier_threshl)
        the_queuel.task_done()
        the_final_queuel.put([outpyyolo,from_queue[4],from_queue[5]])
    pyyolo.cleanup()
    print("tamagno de la cola cuando termino el proceso detector ",integer ," es ", the_queuel.qsize())
    print("El proceso detector",integer,"ve que evento1 termino? ",eventol.is_set())
    if evento2l.is_set()==False:
        evento2l.set()

def build_history_n_annotations(yolo_class_color,yolo_classes_name,evento2l,the_final_queuel):
    count=0
    categories=set()
    storyofclass={}
    while the_final_queuel.qsize()>0 or evento2l.is_set()==False:
        [outputs,frame_id,img_rgb]=the_final_queuel.get()# note that every entry in the_final_queue has the following form [outpyyolo,frame_id,img_rgb]
        if len(outputs)>0:
            for output in outputs:
                if (output["class"] in categories)==True:
                    storyofclass[output["class"]].append(frame_id)   
                else:
                    categories.add(output["class"])
                    storyofclass[output["class"]]=[frame_id]
            annotate_image(img_rgb,yolo_class_color,yolo_classes_name, output_dir=output_dir, detections=outputs, scale=1, sufix="{0:06d}".format(frame_id))   
    json.dump(storyofclass,open(output_file,"w"))


if __name__ == "__main__":
    # yolo_class_color={
    # 'luber_texto':"blue",
    # 'luber_lubri':"blue",
    # 'luber_logo':"blue",
    # 'acdelco_logo':"red",
    # 'acdelco_baterias':"red",
    # 'tablero':"green"}
    # yolo_class_name={
    # 'luber_texto':"Luber",
    # 'luber_lubri':"Luber",
    # 'luber_logo':"Luber",
    # 'acdelco_logo':"ACDelco",
    # 'acdelco_baterias':"ACDelco",
    # 'tablero':"Tablero"}

    yolo_class_color={
    "Head":"blue",
    "Face":"red",
    "Person":"green"
    }
    yolo_class_name={ 
    'Head':0,
    'Face':1,
    'Person':2
    }
    darknet_path = '../'
    data_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/cfg/yolo_metric_train.data'
    cfg_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/cfg/yolo_metric.cfg'
    weight_file = '/mnt/backup/VA/training_arpon/annotations_Head_Face_Person/yolo_metric_train_25000.weights'
    video_file='/mnt/backup/NVR/vivo_coquimbo/cam6/20171001/01000004403000000.mp4'
    output_dir='../results_arpon/annotations_vale_caro/images_video/vivo_coquimbo_cam6_20171001_01000004403000000'
    output_video_file="../results_arpon/vivo_coquimbo_cam6_20171001_01000004403000000.mp4"
    output_video_fps=30
    output_file='../results_arpon/prueba.txt'

    # define initial values
    #frame_id=0
    number_of_detectors=2
    thresh = 0.5
    hier_thresh = 0.5
    stop=0
    dataprev=0

    # Create output folder
    if os.path.isdir(output_dir):
        for file in glob.iglob(os.path.join(output_dir, '*.jpg')):
            os.remove(file)
    else:
        os.makedirs(output_dir, mode=0o777, exist_ok=True)
   
    ts=time.time()

    # logger = mp.log_to_stderr()
    # logger.setLevel(mp.SUBDEBUG)

    #creating the queue
    the_q = mp.JoinableQueue(100)
    the_final_q=mp.Queue()
    lok=mp.Lock()

    #defining an event
    evento=mp.Event()
    evento2=mp.Event()

    # defining capturing process and detector process
    capturing = mp.Process(name="capture",target=capture,args=(number_of_detectors,evento,the_q,video_file))
    detectors=[mp.Process(name="detector "+str(i),target=detect_image,args=(i,lok,evento,evento2,the_q,the_final_q, thresh, hier_thresh,darknet_path, data_file, cfg_file, weight_file)) for i in range(number_of_detectors)]
    bhna=mp.Process(name="bhna",target=build_history_n_annotations,args=(yolo_class_color,yolo_classes_name,evento2,the_final_q))
    
    #starting the detector
    for det in detectors:
        det.start()
    capturing.start()
    bhna.start()


    capturing.join()
    bhna.join()
    count=0    
    for det in detectors:
        print("=============")
        print("contador",count)
        print("El proceso",det.name,"esta vivo? ",det.is_alive())
        print("El proceso", det.name,"ve que el evento ocurrio?",evento.is_set())
        print("El tamagno de la primera cola es ",the_q.qsize())
        print("El tamagno de la segunda cola es ",the_final_q.qsize())
        print("=============")
        if det.is_alive()==True and the_q.qsize()==0 and the_final_q.qsize()==0:
            det.terminate()
        count+=1
    the_q.close()
    the_final_q.close()
    
    #closing the queue
    te=time.time()

    #print(evento.is_set())
    print("total time ",te-ts)



