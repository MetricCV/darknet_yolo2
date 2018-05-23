from convert_bbox_yolo import *
from create_yolo_train import *
from create_yolo_config import *
from intereset_2_bbox import *
import subprocess
# This script creates an instance in darknet from raw annotations and images, to run imediatly from darknet path.
# Input:
# ======
# raw_annpath (str) string which indicates the path to the anotation file
#   Every annotation has the following format " "topx":,"topy":,"height":,"width":,"class":" " 
#   where (topx,topy) correpond to the left top corner of the annotation. y-axes grows in bottoms direction
# raw_imgpath (str) path to the folder where the images are stored (do not add "/" at the end.)
# procc_raw_outputpath (str) path to the folder where the results will be stores will create two new folder in the path, one to store annotations and one to store images (do not add "/" at the end)
# path_2_yolo_template_config (str) folder that contains the yolo_config_template.cfg file
# darknet_path (str) path to darknet_metric
# experiment_name (str) name of the folder saved in darknet_metric/cfg
# backup_dir (str) path where the weights obtained in every iteration will be stored.
# interestclass (list) list with the focus classes
# images_extension (str) indicates the extension of the images (it asummes that all the images have the same extension).
# label_extension (str) indicates the extension of the annotations (it asummes that all the annotations have the same extension).

#Setting Parameters
raw_annotations_path = ""
raw_imgpath = ""
procc_raw_outputpath = ""
path_2_yolo_template_config = ""
darknet_path = ""
experiment_name = ""
backup_dir = ""
interestclass = []
images_extension = ".jpg"
label_extension = ".txt"
yolo_classes = { 
    'head':0,
    'face':1,
    'person':2
    }


if raw_imgpath[-1]=="/":
    raw_imgpath=raw_imgpath[:-1]
if procc_raw_outputpath[-1]=="/":
    procc_raw_outputpath=procc_raw_outputpath[:-1]
if darknet_path[-1]=="/":
    darknet_path=darknet_path[:-1]
            
annotations_of_interest(raw_annotations_path,raw_imgpath,procc_raw_outputpath,interestclass,images_extension,label_extension) # transforming raw annotation in bbox anotations and save it in procc_raw_outputpath
path_images = procc_raw_outputpath+"/images"
path_labels = procc_raw_outputpath+"/labels"
path_yolo = procc_raw_outputpath+"/yolo"
path_cfg = procc_raw_outputpath+"/cfg"
darknet_cfg_experiment = darknet_path+"/cfg"+experiment_name
subprocess.call(["mkdir", path_images])
subprocess.call(["mkdir", path_labels])
subprocess.call(["mkdir", path_yolo])
subprocess.call(["mkdir", path_cfg])
subprocess.call(["mkdir", darknet_cfg_experiment])
mv_images = "*"+images_extension
mv_labels = "*"+label_extension
subprocess.call(["mv", mv_images, path_images])
subprocess.call(["mv", mv_labels, path_labels])
subprocess.call(["mv", path_2_yolo_template_config, path_cfg])
process_files_convert(procc_raw_outputpath, path_yolo, images_extension, label_extension)# converting to yolo format
process_files_create_train(procc_raw_outputpath, path_cfg, images_extension, label_extension)# creating training and test files.
process_files_create_config(path_cfg, path_cfg, backup_dir=backup_dir)# creating configuration
subprocess.call(["cp", "-r", path_cfg , darknet_cfg_experiment])# copying to cfg in darknet_metric folder.