import sys
from pathlib import Path


vit_path = Path(__file__).parent
classify_mnist_path = vit_path.parent
sys.path.append(str(classify_mnist_path))
sys.path.append(str(classify_mnist_path) + '/siren')

import get_train_data
import get_test_data
# import load
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

class CustomMNISTDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        image = item['img']
        label = item['label']
        # img_flatten = item['img_flatten']
        # img_flatten = img_flatten.float().view(1, 28, 28)

        if self.transform:
            image = self.transform(image)
        # img_flatten = img_flatten.view(-1)

        return image, label

mean = 0.1307
std = 0.3081

# transform = transforms.Compose([
#                                 transforms.Normalize((mean, ), (std, )),
# ])

img_df_train, crop_size = get_train_data.make_table()
mnist_train_dataset = CustomMNISTDataset(img_df_train, transform=None)
mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=128, shuffle=True)
# mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=128, shuffle=False, sampler=DistributedSampler(mnist_train_dataset))

img_df_test, _ = get_test_data.make_table()
mnist_test_dataset = CustomMNISTDataset(img_df_test, transform=None)
mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=128, shuffle=True)








# test
# images_flatten, xy_flatten, labels = next(iter(mnist_test_dataloader))

# print(images_flatten.shape)  # (32, 784, 1)
# print(xy_flatten.shape)      # (32, 28*28, 2)
# print(labels.shape)          # (32,)

# img = images_flatten[0].squeeze()
# img = img.reshape(28, 28) 

# plt.imshow(img, cmap='gray')
# plt.savefig('datacheck.jpg')

# print("Labels:", labels[0])



