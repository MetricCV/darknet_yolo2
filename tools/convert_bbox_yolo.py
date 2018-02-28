import os, glob, shutil
import numpy as np
import pandas as pd
from skimage import io
#from PIL import Image

def process_files_convert(input_dir, output_dir,image_ext=".jpg",label_ext=".txt"):

    label_dir="labels"
    image_dir="images"
    # label_ext=".txt"
    # image_ext=".jpg"

    input_label_dir=os.path.join(input_dir, label_dir)
    input_image_dir=os.path.join(input_dir, image_dir)
    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    data_df = search_files(input_dir, label_dir=label_dir, image_dir=image_dir, label_ext=label_ext, image_ext=image_ext)
    print(data_df)
    data_df['label_file_out']=data_df['label_file'].str.replace(input_label_dir, output_dir)
    data_df['image_file_out']=data_df['image_file'].str.replace(input_image_dir, output_dir)

    for i in range(len(data_df)):

        if i % 10 == 0:
            print("Processing file {} of {}".format(i,len(data_df)))

        im_data = io.imread(data_df.loc[i,'image_file'])
        im_width = im_data.shape[1]
        im_height = im_data.shape[0]

        do_parser=parser_file(data_df.loc[i,'label_file'], data_df.loc[i,'label_file_out'], im_width=im_width, im_height=im_height)
        if do_parser:
            shutil.copy(data_df.loc[i,'image_file'], data_df.loc[i,'image_file_out'])

    return data_df

def search_files(input_dir, label_dir="labels", image_dir="images", label_ext=".txt", image_ext=".jpg"):
    image_files=[]
    label_files=[]

    for root, dirs, files in os.walk(os.path.join(input_dir,label_dir)):
        for file in files:
            if file.endswith(label_ext):
                label_file=os.path.join(root, file)
                image_file=os.path.join(os.path.join(input_dir,image_dir), file.replace(label_ext, image_ext))

                label_files.append(label_file)
                image_files.append(image_file)

    df=pd.DataFrame({'label_file':label_files, 'image_file':image_files})
    return(df)

def parser_file(file_in, file_out, im_width=1920, im_height=1080):

    file = open(file_in)
    lines = file.read().split('\n')
    file.close()

    n_obj=int(lines[0])
    lines=lines[1:]

    if n_obj>0:
        file=open(file_out, 'w')
        for line in lines:
            s_line = line.split(" ")
            if len(s_line)==5:
                x_min=s_line[0]
                y_min=s_line[1]
                x_max=s_line[2]
                y_max=s_line[3]
                cls_name=s_line[4]
            
                b = (float(x_min), float(x_max), float(y_min), float(y_max))
                bb = convert_bb((im_width,im_height), b)

                cls_id=yolo_classes[cls_name]

                file.write(str(cls_id) + " " + " ".join(['{:.5f}'.format(a) for a in bb]) + '\n')
        file.close()
        return(True)
    else:
        print("The annotation file doesn't contain objects \n"+file_in)
        return(False)


def convert_bb(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

if __name__ == "__main__":

    # Input:
    # ======
    # -input_dir (str) path to the folder which contain a folder called "images" with "jpg" files and a folder called
    #   "labels" which contains the annotations that contains the annotations in bbox format, which means that
    #   every annotation has the following format " "topx":,"topy":,"height":,"width":,"class":" "
    # -output_dir (str) path to the input_dir and must add /yolo
    # -yolo_classes (dict) the map between Class (which is the key in the dict) and a integer (starting from 0)
    # Output:
    # =======
    # files with annotation in the following format "x_min y_min x_max y_max class" where (x,y)=(0,0) correspond to left 
    #   top corner of the image and (x,y)=(width_image,heigth_image) correspond to the right bottom
    #   corner of the image
    

    # yolo_classes={ 
    # 'luber_texto':0,
    # 'luber_lubri':1,
    # 'acdelco_logo':2,
    # 'luber_logo':3,
    # 'acdelco_baterias':4,
    # 'tablero':5}
    # input_dir="/Volumes/Data/dataset/futbol_mexico"
    # input_dir="/mnt/backup/NVR/futbol_mexico"
    # output_dir="/mnt/backup/NVR/futbol_mexico/yolo"   
    
    yolo_classes={ 
    'Head':0,
    'Face':1,
    'Person':2
    }
    # input_dir="/mnt/data/training_arpon/annotations_Face_Person"
    # output_dir="/mnt/data/training_arpon/annotations_Face_Person/yolo"
    # input_dir="/mnt/data/training_arpon/annotations_Head"
    # output_dir="/mnt/data/training_arpon/annotations_Head/yolo"
    # input_dir="/mnt/data/training_arpon/annotations_Head_Face"
    # output_dir="/mnt/data/training_arpon/annotations_Head_Face/yolo"
    # input_dir="/mnt/data/training_arpon/annotations_Head_Face_Person"
    # output_dir="/mnt/data/training_arpon/annotations_Head_Face_Person/yolo"
    input_dir="/mnt/data/training_arpon/annotations_Head_Person"
    output_dir="/mnt/data/training_arpon/annotations_Head_Person/yolo" 
    df=process_files_convert(input_dir, output_dir)
    print("Convert bbox label annotation to Yolo format is DONE")
