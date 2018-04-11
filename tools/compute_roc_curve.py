import glob
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

def rocgrh(path,output_name,data_path,config_path,stopnum=1000000,darknet_path=".."):
	# This Function read all the files (/*.weights) in a folder, for every file calculates the IOU and RECALL with darknet detector recall
	# and takes the last output of it. After all the files has been used, transform the dict in to a list wich is sorted by epoch.
	#
	# Input:
	# ------
	# -path (str) path to the folder with weights (could be absolute or relative to where the main code is been executed)
	# -darknet_path (str) path to executable file of darknet (could be absolute or relative to where the main code is been executed)
	# -data_path (str) path to file .data (could be absolute or relative to darknet path)
	# -config_path (str) path to file .cfg (could be absolute or relative to darknet path)
	# -output_path (str) path to outputfile (could be absolute or relative to where the main code is been executed)
	# Output:
	# -------
	# -sorted_list_iteration (list) sorted_list_iteration[i] has the following form (epoch (int),"IOU RECALL" (str))
	# -File saved in output_name where the first line is "Epoch         IOU         Recall" and the following
	# 		are the epoch and the associated IOU and Recall, it is sorted by EPOCH
	archiveindir=glob.glob(path+"/*.weights")#getting files in path (path are absolutes)
	iteration={}#dict of epochs
	for i in archiveindir:
		counter=-8 
		stop=0	
		while stop==0:
			counter+=-1	
			if i[counter]=="_":
				stop=1
				numname=i[counter+1:-8]#getting the last number in the file which indicates the epoch
		if int(numname)>stopnum:
			continue
		print(numname)						
		f=open("aux.txt","w")
		subprocess.call(["./darknet", "detector", "recall", data_path, config_path, i],stderr=f,cwd=darknet_path)
		f.close()
		f=open("aux.txt")
		s_line=f.readlines()[-1].split()#reading the last line in the outpufile
		f.close()	
		iteration[float(numname)]=str(float(s_line[6][0:4]))+"    "+str(float(s_line[7][7:10]))#wrting the pair epoch, lastline (IOU RECALL) info
	sorted_list_iteration=[(i,iteration[i]) for i in sorted(iteration.keys())]#sorting iteration by number of epoch.
	f=open(output_name,"w")
	f.write("Epoch         IOU         Recall\n")#writing in outputfile the list sorted_list_iteration
	for i in range(0,len(sorted_list_iteration)):
		f.write(str(sorted_list_iteration[i][0])+"	"+str(sorted_list_iteration[i][1])+"\n")
	f.close()
	os.unlink("aux.txt")# deleting aux file
	return(sorted_list_iteration)

def pltroc(lista_from_rocgrh,graphname):
	# This function takes a list where every entry is a (epoch,IOU,RECALL), and makes a graph of every EPOCH/IOU and EPOCH/RECALL
	#
	# Input:
	# ------
	# -lista_from_rocgrh (list) lista_from_rocgrh[i] has following form (epoch (int),"IOU RECALL" (str))
	# -graphname (str) path to outputgraph (could be absolute or relative to where the main code is been executed) 
	# Output:
	# -------
	# -graph (image) saved in graphname
	iouval=[]
	recallval=[]
	it=[]
	for i in lista_from_rocgrh:
		it.append(int(i[0]))#getting the iteration
		iourec=i[1].split()
		iouval.append(float(iourec[0]))#getting IOU value
		recallval.append(float(iourec[1]))#getting RECALL value
	fig=plt.figure()
	plt.scatter(it,iouval,c='r',label='IOU')
	plt.scatter(it,recallval,c='b',label='Recall')
	plt.xlabel("Iteration")
	plt.ylim([-5,105])
	plt.ylabel("(%)")
	plt.legend(loc='best')
	fig.savefig(graphname)
	
if __name__ == "__main__":
	weight_dir="/mnt/backup/VA/training_arpon/head_face_prioritytag_with_blur_data/"
	roc_data_file="../results_arpon/head_face_prioritytag_with_blur_data_20180405.txt"
	roc_plot_file="../results_arpon/head_face_prioritytag_with_blur_data_20180405.pdf"
	yolo_data_file="cfg/head_face_prioritytag_with_blur_data/yolo_metric_train.data"
	yolo_config_file="cfg/head_face_prioritytag_with_blur_data/yolo_metric.cfg"
	darknet_stop=197000
	darket_dir=".."

	roc_curve=rocgrh(weight_dir, roc_data_file, yolo_data_file, yolo_config_file, stopnum=darknet_stop, darknet_path=darket_dir)
	pltroc(roc_curve, roc_plot_file)