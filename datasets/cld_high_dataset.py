import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import patchify
from torch import torch

# This is where the official pytorch dataset is created
# All it does is make batch training in pytorch much easier
class CLD_high_dataset(Dataset):
    def __init__(self, transforms=None):

        data = np.fromfile("C:\\Users\\7ross\\Desktop\\UK Xin Liang files\\CLDHGH_1_1800_3600.f32", dtype=np.float32)

        data = data.reshape([1800, 3600])
        data = patchify.patchify(data,(72,72),72)
        
        final_list = []
        data = np.squeeze(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                final_list.append(data[i,j,:,:])
        final_list = np.asarray(final_list)
        
        data = torch.from_numpy(final_list)
        self.data = data.unsqueeze(1)
    
    def __getitem__(self, idx):
        return self.data[idx,:,:],0
    
    def __len__(self):
        return self.data.shape[0]
        
# this is the new way to make patches from the original image
def make_jigsaw_puzzle(x, grid_size=(2, 2)):
    # x shape is C x H x W
    C, H, W = x.size()

    assert H % grid_size[0] == 0
    assert W % grid_size[1] == 0

    C, H, W = x.size()
    x_jigsaw = x.unfold(1, H // grid_size[0], W // grid_size[1])
    x_jigsaw = x_jigsaw.unfold(2, H // grid_size[0], W // grid_size[1])
    x_jigsaw = x_jigsaw.contiguous().view(-1, C,  H // grid_size[0], W // grid_size[1])
    return x_jigsaw
# this is the part that patches image input back into the original image
def jigsaw_to_image(x, grid_size=(2, 2)):
    # x shape is batch_size x num_patches x c x jigsaw_h x jigsaw_w
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.size()
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.view(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.view(batch_size, c, output_h, output_w)
    return x_image
