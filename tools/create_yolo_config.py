import os, glob
import re

yolo_classes={ 
    'luber_texto':0,
    'luber_lubri':1,
    'acdelco_logo':2,
    'luber_logo':3,
    'acdelco_baterias':4,
    'tablero':5}

def process_files(input_dir, output_dir, backup_dir="/mnt/backup/VA"):

	yolo_template_file=os.path.join(input_dir, "yolo_template.cfg")

	obj_data_file=os.path.join(output_dir, "yolo_metric_train.data")
	obj_names_file=os.path.join(output_dir, "yolo_metric_train.names")
	yolo_config_file=os.path.join(output_dir, "yolo_metric_train.cfg")
	yolo_detect_config_file=os.path.join(output_dir, "yolo_metric.cfg")

	os.makedirs(output_dir, mode=0o777, exist_ok=True)
	os.makedirs(backup_dir, mode=0o777, exist_ok=True)

	# First, we write the obj_data_file
	text_list = ["classes= {:d}".format(len(yolo_classes)),
		"train  = "+os.path.join(output_dir,"train.txt"),
		"valid  = "+os.path.join(output_dir,"test.txt"),
		"names  = "+os.path.join(output_dir,"yolo_metric_train.names"),
		"backup = "+backup_dir]
	text_list = map(lambda x: x+"\n", text_list)

	file = open(obj_data_file, "w")
	file.writelines(text_list)
	file.close()

	# Second, we write the obj_names_file
	text_list=list(yolo_classes.keys())
	text_list = map(lambda x: x+"\n", text_list)

	file = open(obj_names_file, "w")
	file.writelines(text_list)
	file.close()

	# Third, we read the yolo_template_file and creare the yolo_config_file
	file = open(yolo_template_file, "r")
	lines = file.read().splitlines()
	file.close()

	# We look for lines that begin with a given pattern
	lines.reverse()
	do_filters=True
	for i, line in enumerate(lines):
		if line.startswith("batch="):
			lines[i]="batch=64"
		if line.startswith("subdivisions="):
			lines[i]="subdivisions=8"
		if line.startswith("classes="):
			lines[i]="classes={:d}".format(len(yolo_classes))
		if line.startswith("filters=") and do_filters:
			lines[i]="filters={:d}".format((len(yolo_classes) + 5)*5)
			do_filters=False

	lines.reverse()
	file = open(yolo_config_file, "w")
	for line in lines:
		line=line+"\n"
		file.write(line)

	file.close()


	# We create yolo detect config file
	lines.reverse()
	do_filters=True
	for i, line in enumerate(lines):
		if line.startswith("batch="):
			lines[i]="batch=1"
		if line.startswith("subdivisions="):
			lines[i]="subdivisions=1"
		if line.startswith("classes="):
			lines[i]="classes={:d}".format(len(yolo_classes))
		if line.startswith("filters=") and do_filters:
			lines[i]="filters={:d}".format((len(yolo_classes) + 5)*5)
			do_filters=False

	lines.reverse()
	file = open(yolo_detect_config_file, "w")
	for line in lines:
		line=line+"\n"
		file.write(line)

	file.close()


if __name__ == "__main__":
	input_dir="cfg"
	output_dir="cfg/futbol_mexico"
	backup_dir="/mnt/backup/VA/futbol_mexico/yolo"
	process_files(input_dir, output_dir, backup_dir=backup_dir)