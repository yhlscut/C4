from __future__ import print_function

import numpy as np
import scipy.io
import torch
import torch.utils.data as data

from classes.data.DataAugmenter import DataAugmenter


class ColorCheckerDataset(data.Dataset):

    def __init__(self, train: bool = True, folds_num: int = 1):

        self.train = train
        self.da = DataAugmenter()

        folds = scipy.io.loadmat('./data/folds.mat')
        img_idx = folds['tr_split' if self.train else 'te_split'][0][folds_num][0]

        self.all_data_list = open("./data/color_checker_metadata.txt", 'r').readlines()
        self.data_list = [self.all_data_list[i - 1] for i in img_idx]

    def __getitem__(self, index: int):
        model = self.data_list[index]
        file_name = model.strip().split(' ')[1]
        img = np.array(np.load('./data/ndata/' + file_name + '.npy'), dtype='float32')
        illums = np.array(np.load('./data/nlabel/' + file_name + '.npy'), dtype='float32')

        if self.train:
            img, illums = self.da.augment(img, illums)
        else:
            img = self.da.crop(img)

        img = np.clip(img, 0.0, 65535.0) * (1.0 / 65535)

        # BGR to RGB
        img = img[:, :, ::-1]
        img = np.power(img, (1.0 / 2.2))

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img.copy())
        illums = torch.from_numpy(illums.copy())

        if not self.train:
            img = img.type(torch.FloatTensor)

        return img, illums, file_name

    def __len__(self):
        return len(self.data_list)

# if __name__ == '__main__':
#     dataset = ColorCheckerDataset(train=True)
#     dataload = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=int(30))
#     for ep in range(10):
#         time1 = time.time()
#         for i, data in enumerate(dataload):
#             img, ill, file_name = data
