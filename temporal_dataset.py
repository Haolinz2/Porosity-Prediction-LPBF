import os
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class PorosityDataset(Dataset):    
    def __init__(self, data_dir, time_window=12, use_padding=True, enable_label_history=False):
        self.data_dir = data_dir
        self.time_window = time_window
        self.time_padding = use_padding
        self.enable_label_history = enable_label_history
        
        self.raw_data = defaultdict(dict)
        self.data_indices = []
        for sample_folder in tqdm(os.listdir(data_dir)):
            sample_id = sample_folder[6:]
            for layer_file in os.listdir(os.path.join(data_dir, sample_folder)):
                layer_id = layer_file.split('.')[0]
                mat = loadmat(os.path.join(data_dir, sample_folder, layer_file))

                feat, label = mat['data_cur'], mat['pore_register_cur']
                feat = feat[:, [0, 1, 6, 7, 9, 11, 13]]  # feature selection
                # label[:, 1] = label[:, 1] / label[:, -1]  # normalize the number of pore voxel label
                # label[:, 2] = label[:, 2] / label[:, -1]  # normalize the number of time-windowed pore voxel label
                # label[:, 3] = label[:, 3] / label[:, -1]  # normalize the number of pore label
                label = label[:, :-1]
                self.raw_data[sample_id][layer_id] = (feat, label)

                # create data index
                anchor_i = time_window-1 if not self.time_padding else 0
                for i in range(anchor_i, feat.shape[0]):
                    if np.isnan(label[max(0, i-time_window):i+1]).sum() == 0:  # filtered by valid porosity label
                        self.data_indices.append((sample_id, layer_id, i-time_window+1, i))

    def get_split(self, seed=42):
        # Create random splits
        np.random.seed(seed)
        train_idx = np.random.choice(np.arange(self.__len__()), size=int(0.8*self.__len__()))
        val_idx = np.random.choice(list(set(np.arange(self.__len__())) - set(train_idx)), size=int(0.1*self.__len__()))
        test_idx = np.array(list(set(np.arange(self.__len__())) - set(train_idx) - set(val_idx)))

        # Compute normalization from training features:
        label_array, feat_array, visited = [], [], set()
        for i in train_idx:
            sample_id, layer_id, start_i, end_i = self.data_indices[i]
            feat, label = self.raw_data[sample_id][layer_id]
            for time_id in range(start_i, end_i+1):
                if (sample_id, layer_id, time_id) not in visited:
                    feat_array.append(feat[time_id])
                    label_array.append(label[time_id])
                    visited.add((sample_id, layer_id, time_id))
        feat_array = np.array(feat_array)
        label_array = np.array(label_array)

        self.f_scaler = StandardScaler()
        self.f_scaler.fit(feat_array)

        self.l_scaler = StandardScaler()
        self.l_scaler.fit(label_array[:, 1:])

        return train_idx, val_idx, test_idx
    
    def get_windowed_data(self, idx):
        sample_id, layer_id, start_i, end_i = self.data_indices[idx]
        feats, labels = self.raw_data[sample_id][layer_id]
        
        if start_i < 0:
            feat_i = torch.concat([torch.zeros(-start_i, feats.shape[1]), torch.from_numpy(feats[:end_i+1])]).float()
            label_i = torch.concat([torch.zeros(-start_i).fill_(-1), torch.from_numpy(labels[:end_i+1])]).long()
        else:
            feat_i = torch.from_numpy(feats[start_i:end_i+1]).float()
            label_i = torch.from_numpy(np.array(labels[start_i:end_i+1], float)).long()

        if self.enable_label_history:    
            feat_i = torch.concat([feat_i, label_i.unsqueeze(1)], dim=1)
            feat_i[-1][-1] = -1  # mask out the labeled porosity of the current state

        return feat_i, label_i[-2:-1]
        
    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        feat, label = self.get_windowed_data(idx)
        norm_feat = torch.FloatTensor(self.f_scaler.transform(feat))

        label = torch.concat(
            [label[:, :1], torch.FloatTensor(self.l_scaler.transform(label[:, 1:]))], dim=1)
        return norm_feat, label.squeeze(0)
        

# import os
# import numpy as np
# from scipy.io import loadmat
# from tqdm import tqdm
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler

# import torch
# from torch.utils.data import Dataset


# class PorosityDataset(Dataset):    
#     def __init__(self, data_dir, time_window=12, use_padding=True, enable_label_history=False):
#         self.data_dir = data_dir
#         self.time_window = time_window
#         self.time_padding = use_padding
#         self.enable_label_history = enable_label_history
        
#         self.raw_data = defaultdict(dict)
#         self.data_indices = []
#         for sample_folder in tqdm(os.listdir(data_dir)):
#             sample_id = sample_folder[6:]
#             for layer_file in os.listdir(os.path.join(data_dir, sample_folder)):
#                 layer_id = layer_file.split('.')[0]
                
#                 mat = loadmat(os.path.join(data_dir, sample_folder, layer_file))

#                 feat, label = mat['data_cur'], mat['pore_register_cur']
#                 feat = feat[:, [0, 1, 6, 7, 9, 11, 13]]  # feature selection
#                 self.raw_data[sample_id][layer_id] = (feat, label)

#                 # create data index
#                 anchor_i = time_window-1 if not self.time_padding else 0
#                 for i in range(anchor_i, feat.shape[0]):
#                     if np.isnan(label[max(0, i-time_window):i+1]).sum() == 0:  # filtered by valid porosity label
#                         self.data_indices.append((sample_id, layer_id, i-time_window+1, i))

#     def get_windowed_data(self, idx):
#         sample_id, layer_id, start_i, end_i = self.data_indices[idx]
#         feats, labels = self.raw_data[sample_id][layer_id]
        
#         if start_i < 0:
#             feat_i = torch.concat([torch.zeros(-start_i, feats.shape[1]), torch.from_numpy(feats[:end_i+1])]).float()
#             label_i = torch.concat([torch.zeros(-start_i).fill_(-1), torch.from_numpy(labels[:end_i+1])]).long()
#         else:
#             feat_i = torch.from_numpy(feats[start_i:end_i+1]).float()
#             label_i = torch.from_numpy(np.array(labels[start_i:end_i+1], float)).long()

#         if self.enable_label_history:    
#             feat_i = torch.concat([feat_i, label_i.unsqueeze(1)], dim=1)
#             feat_i[-1][-1] = -1  # mask out the labeled porosity of the current state

#         label_i = label_i[-1]
#         return feat_i, label_i
        
#     def __len__(self):
#         return len(self.data_indices)

#     def __getitem__(self, idx):
#         feat, label = self.get_windowed_data(idx)
#         return feat, label
