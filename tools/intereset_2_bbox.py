import json
import subprocess
import os
def annotations_of_interest(file,imgpath,outputpath,interestclass,img_ext=".jpg",lab_ext=".txt",projectnames=[]):
    # Return files with annotation of interestclass in darknet format for every image
    # Input:
    # ======
    # -path (str) string which indicates the path to the anotation file
    #   Every annotation has the following format " "topx":,"topy":,"height":,"width":,"class":" " 
    #   where (topx,topy) correpond to the left top corner of the annotation. y-axes grows in bottoms direction
    # -imgpath (str) path to the folder where the images are stored
    # -outputpath (str) path to the folder where the results will be stores
    # -interestclass (list) list with the focus classes
    # img_ext (str) image extension
    # lab_ext (str) label extension
    # projetcname (list) if not empty, every entry in projectname is a name of a project.
    # Outputs:
    # =======
    # -files_output (files) one file correspond to the annotations (in interest class) of one image. 
    #   The annotation are in "x_min y_min x_max y_max class" where (x,y)=(0,0) correspond to left 
    #   top corner of the image and (x,y)=(width_image,heigth_image) correspond to the right bottom
    #   corner of the image
    # -copies of the anotated images in outputpath
    rawdata=json.load(open(file,"r"))
    for i in rawdata: #i represents an image
        aux = list(i.keys())
        number_an = 0 # used to count the number of annotation corresponding to interestclass
        if len(projectnames)>0:
            if i["project"] in projectnames:
                for j in aux[:-3]:# do not take in to account "project","image","id" 
                    if type(i[j]) is dict:     
                        print(i[j]["class"]) 
                        if i[j]["class"] in interestclass:
                            number_an += 1
                        else:        
                            try:
                                if (i[j]["type"] in interestclass):
                                    number_an += 1
                            except:
                                pass                      
        if number_an>0:     
            name = outputpath+i["image"]+lab_ext
            inimgpath = imgpath+i["image"]+img_ext
            if os.path.isfile(inimgpath):
                outimgpath = outputpath+i["image"]+img_ext
                subprocess.call(["cp", inimgpath, outimgpath])# moving image in outputpath  
                file = open(name,"w")
                file.write(str(number_an)+"\n")# writing the total number of annotation corresponding to interestclass
                for j in aux[:-3]:# do not take in to account "project","image","id"
                    if i[j]["class"] in interestclass: 
                        max_x = str(int(i[j]["topx"])+int(i[j]["width"]))
                        max_y = str(int(i[j]["topy"])+int(i[j]["height"]))
                        stringline = i[j]["topx"]+" "+i[j]["topy"]+" "+max_x+" "+max_y+" "+i[j]["class"].lower()+"\n"
                        file.write(stringline)
                    elif i[j]["type"] in interestclass:
                        max_x = str(int(i[j]["topx"])+int(i[j]["width"]))
                        max_y = str(int(i[j]["topy"])+int(i[j]["height"]))
                        stringline = i[j]["topx"]+" "+i[j]["topy"]+" "+max_x+" "+max_y+" "+i[j]["type"].lower()+"\n"
                        file.write(stringline)
                file.close()        

if __name__ == '__main__':
    path="/home/sarpon/github/anotaciones cascos y demas/Data_Cruda/an2/anotaciones2.json"
    imgpath="/home/sarpon/github/anotaciones cascos y demas/Data_Cruda/an2/"
    outputpath="/home/sarpon/github/anotaciones cascos y demas/Data_Procesada/annotations"
    projectname=["CSC"]
    # interestclass=["Face","Person"]
    # interestclass=["Head"]
    # interestclass=["Head","Face"]
    # interestclass=["Head","Face","Person"]
    interestclass=["head","person","safety helmet","Truck","forklift truck"]
    for i in interestclass:
        outputpath=outputpath+"_"+i
    outputpath=outputpath+"/"   
    if os.path.isdir(outputpath)==True:
        pass
    else:
        os.mkdir(outputpath)
    annotations_of_interest(path,imgpath,outputpath,interestclass,projectnames=projectname)