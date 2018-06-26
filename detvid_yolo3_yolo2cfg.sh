prin=`date`
./darknet detector demo /mnt/backup/VA/training_arpon/head_face_prioritytag_with_blur_data_with_hats_yolo3_yolo2cfg_20180606/cfg/yolo_metric_train.data  /mnt/backup/VA/training_arpon/head_face_prioritytag_with_blur_data_with_hats_yolo3_yolo2cfg_20180606/cfg/yolo_metric.cfg /mnt/backup/VA/training_arpon/head_face_prioritytag_with_blur_data_with_hats_yolo3_yolo2cfg_20180606/yolo_metric_train_3000.weights /mnt/backup/NVR/vivo_coquimbo/20180430/vivo_coquimbo_192.168.1.103_000_trim.mp4 -i 0
fin=`date`
echo $prin $fin
