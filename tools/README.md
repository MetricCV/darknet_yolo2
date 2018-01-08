# Instructions to setup YOLO for training a custom dataset

## Create configuration files

Edit the files convert_bbox_yolo and create_yolo_config, and modify input_dir and yolo_classes accordingly to your dataset.

Then, execute the scripts in the following order

1. python convert_bbox_yolo.py

2. python create_yolo_train.py

3. python create_yolo_config.py


We need a pretrained convolutional weights file. Download it from https://pjreddie.com/media/files/darknet19_448.conv.23

cd /mnt/backup/dataset/yolo

wget https://pjreddie.com/media/files/darknet19_448.conv.23

## Run YOLO

./darknet detector train yolo/futbol_mexico/yolo_metric.data yolo/futbol_mexico/yolo_metric.cfg /mnt/backup/dataset/yolo/darknet19_448.conv.23 -gpus 0,1