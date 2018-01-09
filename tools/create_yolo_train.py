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


def process_files(input_dir, output_dir):

    label_dir="yolo"
    image_dir="yolo"
    label_ext=".txt"
    image_ext=".jpg"

    os.makedirs(output_dir, mode=0o777, exist_ok=True)

    train_file = os.path.join(output_dir, "train.txt")
    test_file = os.path.join(output_dir, "test.txt")

    data_df = search_files(input_dir, label_dir=label_dir, image_dir=image_dir, label_ext=label_ext, image_ext=image_ext)
    print("data_df: ", len(data_df))

    train_df, test_df = train_test_split(data_df, test_size=0.2)
    print("train_df: ", len(train_df))
    print("test_df: ", len(test_df))
    
    train_df['image_file'].to_csv(train_file, index=False, header=False)
    test_df['image_file'].to_csv(test_file, index=False, header=False)

    return(train_df)


if __name__ == "__main__":
    #input_dir="/Volumes/Data/dataset/futbol_mexico"
    input_dir="/mnt/backup/NVR/futbol_mexico"
    output_dir="cfg/futbol_mexico"
    df=process_files(input_dir, output_dir)
    print(df.head())