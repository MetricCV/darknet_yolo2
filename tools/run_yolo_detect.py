import os, sys, subprocess
from datetime import datetime, timedelta
import pandas as pd

input_dir="/mnt/backup/NVR/futbol_mexico/yolo"
output_dir="/mnt/backup/VA/futbol_mexico/yolo/images"
n_sample=30

yolo_test_file="cfg/futbol_mexico/test.txt"
yolo_data_file="cfg/futbol_mexico/yolo_metric.data"
yolo_config_file="cfg/futbol_mexico/yolo_metric.cfg"
yolo_weight_file="/mnt/backup/VA/futbol_mexico/yolo/yolo_metric_9000.weights"
yolo_thresh=0.25

data=pd.read_csv(yolo_test_file, header=None)
data.columns=['file_in']
data['file_out']=data['file_in'].str.replace(input_dir, output_dir)

data_sample=data.sample(n=n_sample).reset_index(drop=True)

time_start = datetime.now()
for i in range(len(data_sample)):
    
    command=[ "./darknet", "detector", "test",
        yolo_data_file, yolo_config_file, yolo_weight_file,
        data_sample.loc[i,"file_in"], "-thresh", "{:.2f}".format(yolo_thresh),
        "-out", data_sample.loc[i,"file_out"] ]

    print(command)
    try:
        command_output = subprocess.check_output(command, cwd="../", stderr=subprocess.STDOUT)
    except CalledProcessError as exc:
        print("darknet FAILED: ", exc.returncode, exc.output)
    else:
        log = command_output

time_end = datetime.now()
print("Total execution time: ", (time_end-time_start).total_seconds())