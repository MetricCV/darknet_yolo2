import glob
from shapely.geometry import box
import glob
import subprocess
import shutil


def create_art_cat(folder,threshold_percentage):
    # This fuctions assumes that all the annotations has the following three categories person,head and safety helmet. It assumes darknet format. Note that the category safety helmet will be deleted from every annotation
    # This function take the heads and safety helmet categories and transform it in to head with helmet and heads without helmet (if one helmet is alone)
    # Inputs:
    # =======
    # folder (str) folder with the annotations.
    # Outputs:
    # ========
    # output new annotations files with the category head replaced by head with helmet and head without helmet.
    yolo_classes={ 
    'head':0,
    'person':1,
    'safety helmet':2
    }

    yolo_classes_after={ 
    'head_woh':0,
    'person':1,
    'safety helmet':2,
    'head_wh':3
    }

    if folder[-1]=="/":
        folder[-1]==""
    ann_list=glob.glob(folder+"/*.txt")
    print(ann_list)
    for i in ann_list:
        f=open(i,"r")
        heads=[]
        helmets=[]
        person=[]
        for line in f.read().splitlines(): # every line has the following format "class porcentual_x_center porcentual_y_center porcentual_width porcentual_height"
            # line=line.split()
            if line[0]=="0":
                heads.append(line)
            elif line[0]=="1":
                person.append(line)    
            elif line[0]=="2":
                helmets.append(line)
        # return(heads)
        f.close()
        heads=[[float(i) for i in j.split()] for j in heads]
        helmets=[[float(i) for i in j.split()] for j in helmets]
        person=[[float(i) for i in j.split()] for j in person]
        filtered_head=list(heads)
        filtered_helmets=list(helmets)
        # filtered_helmets=list(helmets)
        if len(helmets)==0:
            shutil.copy2(i,folder+"/retag/"+i.split("/")[-1])    
            shutil.copy2(i.replace(".txt",".jpg"),folder+"/retag/"+i.split("/")[-1].replace(".txt",".jpg"))
        elif len(helmets)>0:
            if len(heads)>0: # the only way that this will not be is that len(heads)=0:
                for j in helmets:
                    helmet2head=[] # list which links face with one head trought max of IOU
                    hmet_box=box((j[1]-(j[3]/2.0)),(j[2]+(j[4]/2.0)),(j[1]+(j[3]/2.0)),(j[2]-(j[4]/2.0)))
                    head_boxes=[]
                    for k in heads:
                        head_box=box((k[1]-(k[3]/2.0)),(k[2]+(k[4]/2.0)),(k[1]+(k[3]/2.0)),(k[2]-(k[4]/2.0)))
                        helmet2head.append(float(hmet_box.intersection(head_box).area)/float(hmet_box.union(head_box).area))
                    print(helmet2head)
                    helmet2head_index=helmet2head.index(max(helmet2head)) #this indicates the k-th index in the heads list can be changed to 2
                    if float(helmet2head[helmet2head_index])>threshold_percentage:
                        filtered_head[helmet2head_index][0]=3
                        filtered_helmets.remove(j)                      
                filtered_head=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in filtered_head]
                filtered_helmets=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in filtered_helmets]
                person=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in person]
                file=open(folder+"/retag/"+i.split("/")[-1],"w")
                file.writelines(filtered_head)
                file.writelines(filtered_helmets)
                file.writelines(person)
                file.close()
                shutil.copy2(i.replace(".txt",".jpg"),folder+"/retag/"+i.split("/")[-1].replace(".txt",".jpg"))
            else:
                filtered_helmets=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in filtered_helmets]
                person=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in person]
                file=open(folder+"/retag/"+i.split("/")[-1],"w")
                file.writelines(filtered_helmets)
                file.writelines(person)
                file.close()
                shutil.copy2(i.replace(".txt",".jpg"),folder+"/retag/"+i.split("/")[-1].replace(".txt",".jpg"))   
if __name__ == '__main__':
    folder="/mnt/data/training_arpon/annotations_head_person_helmet/yolo/"
    create_art_cat(folder,0)