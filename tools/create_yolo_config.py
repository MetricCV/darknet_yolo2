import os, glob

yolo_classes={ 
    'luber_texto':0,
    'luber_lubri':1,
    'acdelco_logo':2,
    'luber_logo':3,
    'acdelco_baterias':4}

def process_files(input_dir, output_dir, backup_dir="/mnt/backup/VA"):

	yolo_template_file=os.path.join(input_dir, "yolo_template.cfg")

	obj_data_file=os.path.join(output_dir, "yolo_metric.data")
	obj_names_file=os.path.join(output_dir, "yolo_metric.names")
	yolo_config_file=os.path.join(output_dir, "yolo_metric.cfg")

	os.makedirs(output_dir, mode=0o777, exist_ok=True)
	os.makedirs(backup_dir, mode=0o777, exist_ok=True)

	# First, we write the obj_data_file
	text_list = ["classes= {:d}".format(len(yolo_classes)),
		"train  = "+os.path.join(output_dir,"train.txt"),
		"valid  = "+os.path.join(output_dir,"test.txt"),
		"names  = "+os.path.join(output_dir,"yolo_metric.names"),
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
	lines = file.read().split('\n')
	file.close()

	file = open(yolo_config_file, "w")
	for line in lines:

		if line.startswith("batch="):
			line="batch=64"
		if line.startswith("subdivisions="):
			line="subdivisions=8"
		if line.startswith("classes="):
			line="classes={:d}".format(len(yolo_classes))
		if line.startswith("filters="):
			line="filters={:d}".format((len(yolo_classes) + 5)*5)

		line=line+"\n"
		file.write(line)

	file.close()


if __name__ == "__main__":
	input_dir="yolo"
	output_dir="cfg/futbol_mexico"
	backup_dir="/mnt/backup/VA/futbol_mexico/yolo"
	process_files(input_dir, output_dir, backup_dir=backup_dir)