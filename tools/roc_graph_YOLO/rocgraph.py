import glob
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
def rocgrh(path,output_name):
	archiveindir=glob.glob(path+"/*.weights")#getting files in path
	iteration={}
	for i in archiveindir:
		counter=-8
		stop=0	
		while stop==0:
			counter+=-1	
			if i[counter]=="_":
				stop=1
				numname=i[counter+1:-8]#getting the last number in the file
				print(numname)					
		f=open("aux.txt","w")
		subprocess.call(["./darknet", "detector", "recall", "cfg/futbol_mexico/yolo_metric.data", "cfg/futbol_mexico/yolo_metric.cfg",i],stderr=f,cwd="..")
		f.close()
		f=open("aux.txt")
		s_line=f.readlines()[-1].split()#reading the last line in the outpufile
		f.close()	
		iteration[float(numname)]=str(float(s_line[6][0:4]))+"    "+str(float(s_line[7][7:10]))#wrting the pair iteration, lastline (IOU RECALL) info
	sorted_list_iteration=[(i,iteration[i]) for i in sorted(iteration.keys())]#sorting iteration by number of iteration.
	f=open(output_name,"w")
	f.write("iteration	IOU	Recall\n")#writing in outputfile the list sorted_list_iteration
	for i in range(0,len(sorted_list_iteration)):
		f.write(str(sorted_list_iteration[i][0])+"	"+str(sorted_list_iteration[i][1])+"\n")
	f.close()
	return(sorted_list_iteration)
def pltroc(lista_from_rocgrh,graphname):
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
	plt.scatter(it,recallvall,c='b',label='Recall')
	plt.xlabel("Iteration")
	plt.ylabel("(%)")
	plt.legend(loc='best')
	fig.savefig(graphname)#note that we are saving the figure in the same path in which we are running the ./darknet
	
