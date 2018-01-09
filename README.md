# Instructions to setup YOLO for training a custom dataset

## Create configuration files

Open the folder tools

Edit the files convert_bbox_yolo and create_yolo_config.

Modify the variables input_dir, output_dir and yolo_classes according to your dataset.

Execute the following scripts in this order

1. `python convert_bbox_yolo.py`

2. `python create_yolo_train.py`

3. `python create_yolo_config.py`

Once you execute the previous scripts, you should have the configuration files inside the folder cfg. Copy the contents into the root cfg folder.

`cp -r cfg/futbol_mexico ../cfg`

## Download pretrained convolutional weight file

We need a pretrained convolutional weights file. Download it from https://pjreddie.com/media/files/darknet19_448.conv.23

cd /mnt/backup/dataset/yolo

wget https://pjreddie.com/media/files/darknet19_448.conv.23

## Run YOLO training

./darknet detector train cfg/futbol_mexico/yolo_metric.data cfg/futbol_mexico/yolo_metric.cfg /mnt/backup/dataset/yolo/darknet19_448.conv.23 -gpus 0,1

## Restart YOLO training from partial result

./darknet detector train cfg/futbol_mexico/yolo_metric.data cfg/futbol_mexico/yolo_metric.cfg /mnt/backup/VA/futbol_mexico/yolo/yolo_metric_9000.weights -gpus 0,1

## Compute the recall ratio from YOLO

./darknet detector recall cfg/futbol_mexico/yolo_metric.data cfg/futbol_mexico/yolo_metric.cfg /mnt/backup/VA/futbol_mexico/yolo/yolo_metric_9000.weights

## Detect objects in an image using the trained YOLO model

./darknet detector test cfg/futbol_mexico/yolo_metric.data cfg/futbol_mexico/yolo_metric.cfg /mnt/backup/VA/futbol_mexico/yolo/yolo_metric_9000.weights /mnt/backup/NVR/futbol_mexico/yolo/futbol_mexico_00_30_57_img0052.jpg -thresh 0.25 -out /mnt/backup/VA/futbol_mexico/yolo/images/futbol_mexico_00_30_57_img0052.jpg
