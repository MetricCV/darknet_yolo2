import json
import shutil
import glob
from pycocotools.coco import COCO
def take_vehicules(path_with_images_train,path_with_images_val,darknet_annotations_folder_train,darknet_annotations_folder_val,json_file,output_dir,class_needed=[3,4,8]):
	# This function look for the annotations (darknet format) of coco an rescue all the images where 
	# the annotations contains only classes of class_needed
	# Inputs:
	# =======
	# path_with_images_train (str) path to the folder with train images
	# path_with_images_val (str) path to the folder with validation images
	# darknet_annotations_folder_train (str) path annotation in darknet format train 
	# darknet_annotations_folder_val (str) path annotation in darknet format validation 
	# json_file (str) json with all the annotations
	# output_dir (str) folder where the image and annotation will be stored
	# class_needed (list) names of the desired classes (must be in english and lower case letters)
	# Outputs:
	# ========
	if output_dir[-1]=="/":
		output_dir[-1]==""
	if path_with_images_train[-1]=="/":
		path_with_images_train[-1]=""
	if path_with_images_val[-1]=="/":
		path_with_images_val[-1]=""
	if darknet_annotations_folder_train[-1]=="/":
		darknet_annotations_folder_train[-1]=""
	if darknet_annotations_folder_val[-1]=="/":
		darknet_annotations_folder_val[-1]=""
	coco=COCO(json_file)
	img_ids=coco.getImgIds(catIds=class_needed) # this are all the image ids which contains any class in catIds
	print(img_ids)
	filtered_images=img_ids
	# for i in img_ids:
	# 	next_image=False
	# 	ann=coco.loadAnns(coco.getAnnIds(imgIds=i)) # this are all the anotations that are in image i
	# 	for j in ann:
	# 		if j["category_id"] in class_needed: 
	# 			continue
	# 		else: # if one 
	# 			next_image=True
	# 			break	
	# 	if next_image==False:
	# 		filtered_images.append(i)	
	for i in filtered_images:
		image_file1=path_with_images_train+"/"+coco.imgs[i]['file_name']
		image_file2=path_with_images_val+"/"+coco.imgs[i]['file_name']
		text_annotation1=darknet_annotations_folder_train+"/"+coco.imgs[i]['file_name'].replace(".jpg",".txt")	
		text_annotation2=darknet_annotations_folder_val	+"/"+coco.imgs[i]['file_name'].replace(".jpg",".txt")
		outimg=output_dir+"/"+coco.imgs[i]['file_name']
		outann=output_dir+"/"+coco.imgs[i]['file_name'].replace(".jpg",".txt")
		try:
			shutil.copy2(image_file1,outimg)
		except:
			shutil.copy2(image_file2,outimg)
		try:
			shutil.copy2(text_annotation1,outann)
		except:
			shutil.copy2(text_annotation2,outann)

if __name__ == '__main__':
	path_with_images_train="/mnt/backup/dataset/coco/images/train2014"
	path_with_images_val="/mnt/backup/dataset/coco/images/val2014"
	darknet_annotations_folder_train="/mnt/backup/dataset/coco/labels/train2014"
	darknet_annotations_folder_val="/mnt/backup/dataset/coco/labels/train2014"
	json_file="/mnt/backup/dataset/coco/annotations/instances_train2014.json"
	output_dir="/mnt/data/training_arpon/annotations_head_person_helmet_truck_car_motorcycle_bycicle/coco_truck_motorcycle_car/"		
	take_vehicules(path_with_images_train,path_with_images_val,darknet_annotations_folder_train,darknet_annotations_folder_val,json_file,output_dir)


