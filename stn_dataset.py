import glob
import os
import pathlib

import cv2
import numpy as np
from torch.utils.data import Dataset


class STNDataset(Dataset):

    def __init__(self, root, dataset_type="train", gamma=1.5, transform=None, target_transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data = self._read_data()
        self.min_image_num = -1
        self.gamma = gamma

        self.class_stat = None

    def _getitem(self, index):
        image_info = self.data[index]
        x_image = self._read_image(image_info['x_train'])
        y_image = self._read_image(image_info['y_train'])
        if self.transform:
            x_image = self.transform(x_image)
        if self.target_transform:
            y_image = self.transform(y_image)
        return x_image, y_image

    def __getitem__(self, index):
        x_image, y_image = self._getitem(index)
        return x_image, y_image

    def _read_data(self):
        x_train = glob.glob(str(self.root) + '/' + self.dataset_type + '/*.jpg')
        data = []
        for f in x_train:
            data.append({
                'x_train': os.path.basename(f),
                'y_train': str(os.path.basename(f.rstrip('.jpg') + '-syn.png'))
            })
        return data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}"]
        return "\n".join(content)

    def _adjust_gamma(self, image):
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / image_id
        image = cv2.imread(str(image_file))
        image = self._adjust_gamma(image)

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
