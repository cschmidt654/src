
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import patchify


# This is where the official pytorch dataset is created
# All it does is make batch training in pytorch much easier


class CLD_high_dataset(Dataset):
    def __init__(self, transforms=None):

        data = np.fromfile("C:\\Users\\7ross\\Desktop\\UK Xin Liang files\\CLDHGH_1_1800_3600.f32", dtype=np.float32)

        data = data.reshape([1800, 3600])
        data = patchify.patchify(data,(32,32),32)
        print(data.shape)
        final_list = []
        data = np.squeeze(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                final_list.append(data[i,j,:,:])
        final_list = np.asarray(final_list)
        print(final_list.shape)
        self.data= final_list
    
    def __getitem__(self, idx):
        return self.data[idx,:,:]
    
    def __len__(self):
        return self.data[0]
        
        
dataset = CLD_high_dataset()
    
   