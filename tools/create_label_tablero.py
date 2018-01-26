import os
import sys
import glob

im_dir="/mnt/backup/NVR/futbol_mexico/images_nologo"

'''
if os.path.isdir(im_dir):
	im_files=glob.glob(os.path.join(im_dir,"*.jpg"))
	label_files = list(map(lambda x: x.replace(".jpg",".txt"), im_files))

	for label_file in label_files:
		f=open(label_file, "w")
		f.write("1\n")
		f.write("201 98 590 154 tablero\n")
		f.close()


label_dir="/mnt/backup/NVR/futbol_mexico/labels"

if os.path.isdir(label_dir):
	label_files=glob.glob(os.path.join(label_dir,"futbol_mexico_0*.txt"))
	#print(label_files)

	for label_file in label_files:
		f=open(label_file, "r")
		lines = f.read().splitlines()
		f.close()

		if len(lines)>1:
			f=open(label_file, "w")
			for i in range(len(lines)):
				if i==0:
					new_line=str(int(lines[i])+1)
				else:
					new_line=lines[i]

				f.write(new_line+"\n")

			f.write("201 98 590 154 tablero\n")
			f.close()
'''
