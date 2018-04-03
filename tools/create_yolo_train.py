import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split

def search_files(input_dir, label_dir="yolo", image_dir="yolo", label_ext=".txt", image_ext=".jpg"):
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


def process_files_create_train(input_dir, output_dir,image_ext=".jpg",label_ext=".txt"):

    label_dir="yolo"
    image_dir="yolo"
    # label_ext=".txt"
    # image_ext=".jpg"

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    train_file = os.path.join(output_dir, "train.txt")
    test_file = os.path.join(output_dir, "test.txt")

    data_df = search_files(input_dir, label_dir=label_dir, image_dir=image_dir, label_ext=label_ext, image_ext=image_ext)
    porcentual=min(0.2,1000.0/len(data_df))
    train_df, test_df = train_test_split(data_df, test_size=porcentual)
    print("train_df: ", len(train_df))
    print("test_df: ", len(test_df))
    
    train_df['image_file'].to_csv(train_file, index=False, header=False)
    test_df['image_file'].to_csv(test_file, index=False, header=False)

    return(train_df)


if __name__ == "__main__":
    # Input:
    # ======
    # -input_dir (str) path to the folder which contain a folder called "images" with "jpg" files and a folder called
    #   "labels" which contains the annotations that contains the annotations in bbox format, which means that
    #   every annotation has the following format " "topx":,"topy":,"height":,"width":,"class":" "
    # -output_dir (str) path to the folder where the results will be saved
    # Output:
    # =======
    # 

    #input_dir="/Volumes/Data/dataset/futbol_mexico"
    # input_dir="/mnt/backup/NVR/futbol_mexico"
    # output_dir="cfg/futbol_mexico"
    # input_dir="/mnt/data/training_arpon/annotations_Face_Person"
    # output_dir="cfg/annotations_Face_Person"
    # input_dir="/mnt/data/training_arpon/annotations_Head"
    # output_dir="cfg/annotations_Head"
    # input_dir="/mnt/data/training_arpon/annotations_Head_Face"
    # output_dir="cfg/annotations_Head_Face"
    # input_dir="/mnt/data/training_arpon/annotations_Head_Face_Person"
    # output_dir="cfg/annotations_Head_Face_Person"
    # input_dir="/mnt/data/training_arpon/annotations_Head_Person"
    # output_dir="cfg/annotations_Head_Person" 
    input_dir="/mnt/data/head_face_prioritytag_with_blur_data/agregado"
    output_dir="cfg/head_face_prioritytag_with_blur_data"
    df=process_files_create_train(input_dir, output_dir)
    print(df.head())