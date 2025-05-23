from tqdm import tqdm
import subprocess
import pandas as pd 
import os 
import sklearn.model_selection
from sklearn.model_selection import train_test_split
import glob


def split_img_label(data_train,data_test,folder_train,folder_test):
    
    # os.mkdir(folder_train)
    # os.mkdir(folder_test)
    main_dir = "dataset"
    sub_dirs = ["images", "labels"]
    train_val_dirs = ["train", "val"]
    
    # Create the main dataset directory
    os.makedirs(main_dir, exist_ok=True)
    
    # Create subdirectories and train/val folders
    for sub_dir in sub_dirs:
        # Create the folder for images and labels
        sub_dir_path = os.path.join(main_dir, sub_dir)
        os.makedirs(sub_dir_path, exist_ok=True)
        
        for train_val in train_val_dirs:
            # Create the train and val folders inside images and labels
            os.makedirs(os.path.join(sub_dir_path, train_val), exist_ok=True)

    
    train_ind=list(data_train.index)
    test_ind=list(data_test.index)
    folder_train_images="dataset\\images\\train"
    folder_train_labels="dataset\\labels\\train"
    folder_test_images="dataset\\images\\val"
    folder_test_labels="dataset\\labels\\val"
  
    # Train folder
    for i in tqdm(range(len(train_ind))):
        source ='.\\'+ "tmp\\images" + '\\'  +data_train[train_ind[i]].split('/')[3]
        destination = '.\\'+ folder_train_images + '\\'  +data_train[train_ind[i]].split('/')[3]
        command = f'copy "{source}" "{destination}"'
        os.system(command)
        source ='.\\'+ "tmp\\labels" + '\\'  +data_train[train_ind[i]].split('/')[3].split('.jpg')[0]+'.txt'
        destination = '.\\'+ folder_train_labels + '\\'  +data_train[train_ind[i]].split('/')[3].split('.jpg')[0]+'.txt'
        command = f'copy "{source}" "{destination}"'
        os.system(command)
     
# Execute the command
#     Test folder
    for j in tqdm(range(len(test_ind))):
        source ='.\\'+ "tmp\\images" + '\\'  +data_test[test_ind[j]].split('/')[3]
        destination = '.\\'+ folder_test_images + '\\'  +data_test[test_ind[j]].split('/')[3]
        command = f'copy "{source}" "{destination}"'
        os.system(command)
        source ='.\\'+ "tmp\\labels" + '\\'  +data_test[test_ind[j]].split('/')[3].split('.jpg')[0]+'.txt'
        destination = '.\\'+ folder_test_labels + '\\'  +data_test[test_ind[j]].split('/')[3].split('.jpg')[0]+'.txt'
        command = f'copy "{source}" "{destination}"'
        os.system(command)
def apply_split():
    PATH_Images = './tmp/images/'
    PATH_Labels= './tmp/labels/'
    
    list_img=[img for img in os.listdir(PATH_Images) if img.endswith('.jpg')==True]# you can change this extension to jpg
    list_txt=[img for img in os.listdir(PATH_Labels) if img.endswith('.txt')==True]
    
    path_img=[]
    
    for i in range (len(list_img)):
        path_img.append(PATH_Images+list_img[i])
        
    df=pd.DataFrame(path_img)
    
    # split 
    data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=0.20, random_state=42)
    folder_train_name="train"
    folder_test_name="test"
    # Function split 
    split_img_label(data_train,data_test,folder_train_name,folder_test_name)