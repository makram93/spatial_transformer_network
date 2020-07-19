import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):
    CHECKPOINT_FILENAME_PATTERN = 'STN_model-Epoch-{}.pth'

    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(32 * 1 * 7, 1024)
        self.fc2 = nn.Linear(1024, 6)

        # Initialize the weights/bias with identity transformation
        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, xs):
        # transform the input
        x = F.relu(self.conv1(xs))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 1 * 7)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 3)  # change it to the 2x3 matrix

        grid = F.affine_grid(x, xs.size(), align_corners=True)
        x = F.grid_sample(xs, grid, align_corners=True)

        return x

    def store(self, path_to_dir, step, maximum=5):
        path_to_models = glob.glob(os.path.join(path_to_dir, STN.CHECKPOINT_FILENAME_PATTERN.format('*')))

        if len(path_to_models) == maximum:
            min_step = min([int(path_to_model.split('/')[-1][20:-4]) for path_to_model in path_to_models])
            path_to_min_step_model = os.path.join(path_to_dir, STN.CHECKPOINT_FILENAME_PATTERN.format(min_step))
            os.remove(path_to_min_step_model)

        path_to_checkpoint_file = os.path.join(path_to_dir, STN.CHECKPOINT_FILENAME_PATTERN.format(step))
        torch.save(self.state_dict(), path_to_checkpoint_file)
        return path_to_checkpoint_file

    def restore(self, path_to_checkpoint_file):
        self.load_state_dict(torch.load(path_to_checkpoint_file))
        step = int(path_to_checkpoint_file.split('/')[-1][6:-4])
        return step
