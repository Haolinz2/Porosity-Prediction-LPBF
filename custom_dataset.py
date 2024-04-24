import os
import torch
import numpy as np
import torch.nn as nn
import glob
import pathlib
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
import torch
from scipy.io import loadmat
### define the classes and concatenate the classes together


class PorosityDataset(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir,transform=None):
        
        # Setup transforms
        self.paths_a = glob.glob(targ_dir + '/*/*/area/*')
        self.paths_w = glob.glob(targ_dir + '/*/*/width/*')
        self.paths_l = glob.glob(targ_dir + '/*/*/length/*')
        self.paths_t = glob.glob(targ_dir + '/*/*/temp/*')
        self.paths_i = glob.glob(targ_dir + '/*/*/intensity/*')
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int):
        "Opens an image via a path and returns it."
        image_path_a = self.paths_a[index]
        image_path_w = self.paths_w[index]
        image_path_l = self.paths_l[index]
        image_path_t = self.paths_t[index]
        image_path_i = self.paths_i[index]

        return Image.open(image_path_t), Image.open(image_path_i), Image.open(image_path_a), Image.open(image_path_w), Image.open(image_path_l)
    
    ### return the length of the dataset
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths_a)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img_t,img_i,img_a,img_w, img_l = self.load_image(index)
        class_name  = os.path.split(Path(self.paths_a[index]).parents[2])[1] # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]
        array_t = np.resize(np.array(img_t),(12,7,1))
        array_i = np.resize(np.array(img_i),(12,7,1))
        array_a = np.resize(np.array(img_a),(12,7,1))
        array_w = np.resize(np.array(img_w),(12,7,1))
        array_l = np.resize(np.array(img_l),(12,7,1))
        array_images = np.concatenate((array_t, array_i, array_a,array_w,array_l), axis = 2)
        # Transform if necessary
        if self.transform:
            return self.transform(array_images),class_idx # return data, label (X, y)
        else:
            return array_images, class_idx # return data, label (X, y)

class PorosityDataset_mat(Dataset):

    def __init__(self, targ_dir,transform=None):
        
        # Setup transforms
        self.paths = glob.glob(targ_dir + '/*/*.mat')
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    
    ### return the length of the dataset
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        cur_data = torch.tensor(loadmat(self.paths[index])['cur_scalogram']).permute((0, 1, 2)).contiguous()
        class_name  = os.path.split(Path(self.paths[index]).parents[0])[1] # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(cur_data),class_idx # return data, label (X, y)
        else:
            return cur_data, class_idx # return data, label (X, y)

class PorosityDataset_mat_sigs(Dataset):

    def __init__(self, targ_dir_list,transform=None):
        
        # Setup transforms
        self.paths = targ_dir_list
        self.transform = transform
        # Create classes and class_to_idx attributes

    
    ### return the length of the dataset
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        cur_data = torch.tensor(loadmat(self.paths[index])['cur_scalogram']).permute((0, 1, 2)).contiguous()
        cur_label = loadmat(self.paths[index])['cur_label'].T
        cur_label = cur_label.astype(np.float32)
        # Transform if necessary
        if self.transform:
            return self.transform(cur_data),cur_label # return data, label (X, y)
        else:
            return cur_data, cur_label # return data, label (X, y)




def find_classes(directory: str):


    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    #
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
