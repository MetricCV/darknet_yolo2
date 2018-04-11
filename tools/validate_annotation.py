import matplotlib
matplotlib.use('Agg') 
from shapely.geometry import box
import glob
import subprocess
import shutil
import cv2
import matplotlib.pyplot as plt
def validate_annotation(folder,min_width=40, min_height=40,threshold_percentage=0.25):
    # This function takes annotation of images in darknet format and filter which correspond to head and which correspond to face.
    # The filtered annotation, and the corresponding image, is saved in /yolo folder with darknet format.
    # Inputs:
    # ======
    # folder (str) path to the folder with the images and the annotations (both in the same folder)
    # min_width (int) minimum width of a face
    # min_height (int) minimum height of a face
    # Output:
    # =======
    # Filtered annotation, and the corresponding image, is saved in /yolo folder with darknet format
    # box(float(i["left"]),float(i["top"]),float(i["right"]),float(i["bottom"]))
    # inter_area=float(f_box.intersection(h_box).area)
    # union_area=float(f_box.union(h_box).area)
    # IOU[i,j]=inter_area/union_area
    if folder[-1]=="/":
        folder[-1]==""
    annotations_list=glob.glob(folder+"/*.txt")
    print(annotations_list)
    for i in annotations_list:
        f=open(i,"r")
        heads=[]
        faces=[]
        for line in f.read().splitlines(): # every line has the following format "class porcentual_x_center porcentual_y_center porcentual_width porcentual_height"
            # line=line.split()
            if line[0]=="0":
                heads.append(line)
            elif line[0]=="1":
                faces.append(line)
        # return(heads)
        heads=[[float(i) for i in j.split()] for j in heads]
        faces=[[float(i) for i in j.split()] for j in faces]
        filtered_head=list(heads)
        filtered_faces=list(faces)
        if len(heads)==0 or len(faces)==0:
            shutil.copy2(i,folder+"/yolo/"+i.split("/")[-1])    
            shutil.copy2(i.replace(".txt",".jpg"),folder+"/yolo/"+i.split("/")[-1].replace(".txt",".jpg"))
        else:
            h,w,c = cv2.imread(i.replace(".txt",".jpg")).shape
            for j in faces:
                if j[3]*w>=min_width and j[4]*h>=min_height: # if the face is wide and high enough
                    face2head=[] # list which links face with one head trought max of IOU
                    f_box=box((j[1]-(j[3]/2.0))*w,(j[2]+(j[4]/2.0))*h,(j[1]+(j[3]/2.0))*w,(j[2]-(j[4]/2.0))*h)
                    x1,y1=f_box.exterior.xy
                    head_boxes=[]
                    for k in heads:
                        h_box=box((k[1]-(k[3]/2.0))*w,(k[2]+(k[4]/2.0))*h,(k[1]+(k[3]/2.0))*w,(k[2]-(k[4]/2.0))*h)
                        # head_boxes.append(h_box)
                        face2head.append(float(f_box.intersection(h_box).area)/float(f_box.union(h_box).area))
                    face2head_index=face2head.index(max(face2head))
                    if float(face2head[face2head_index])<=threshold_percentage:
                        filtered_faces.remove(j)
                    else:
                        try:
                            filtered_head.remove(heads[face2head_index])
                        except:
                            pass         
                else:
                    filtered_faces.remove(j)
            filtered_faces=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in filtered_faces]
            filtered_head=[str(int(i[0]))+" "+str(i[1]) +" "+str(i[2])+" "+str(i[3]) +" "+str(i[4])+"\n" for i in filtered_head]
            file=open(folder+"/yolo/"+i.split("/")[-1],"w")
            file.writelines(filtered_head)
            file.writelines(filtered_faces)
            file.close()
            shutil.copy2(i.replace(".txt",".jpg"),folder+"/yolo/"+i.split("/")[-1].replace(".txt",".jpg"))

if __name__ == '__main__':
    folder="/mnt/data/head_face_prioritytag_with_blur_data/anotacion_gorros/"
    validate_annotation(folder,min_width=40, min_height=40,threshold_percentage=0.15)    
        

