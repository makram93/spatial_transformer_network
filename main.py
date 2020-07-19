import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from model import STN
from stn_dataset import STNDataset
from utils import visualize_stn

parser = argparse.ArgumentParser(description='SVHN Regression Training With Pytorch')

parser.add_argument('--dataset_path', type=str, default='data/stn_data', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--train_fraction', default=0.90, type=float,
                    help='Fraction of total data set to be used for Training and remaining for validation')
parser.add_argument('--log_path', type=str, default='logs', help='Log directory path')
parser.add_argument('--num_workers', type=int, default=4, help='num workers')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0., help='momentum')
parser.add_argument('--weight_decay', type=float, default=0., help='weight_decay')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--visualise', type=bool, default=False, help='Visulaize results')
parser.add_argument('--checkpoint_dir', type=str, default='ckp/', help='path to store checkpoints')
parser.add_argument('--restore_ckp', type=str, default='ckp/', help='path to restore checkpoints')
parser.add_argument('--validation_epochs', type=int, default=2, help='epochs after which validation is performed')
parser.add_argument('--debug_steps', type=int, default=10, help='steps after which loss should be printed')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
args = parser.parse_args()
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(data_loader, net, optimizer, device, ep=0):
    model.train()
    running_loss = []
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = F.l1_loss(output, target) + F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
    logging.info(
        f"Epoch: {ep + 1}, " +
        f"Training Loss: {np.mean(running_loss):.4f}"
    )
    if (epoch + 1) % 2 == 0 and args.visualise:
        visualize_stn(model, test_loader, device)


def test(loader, net, device):
    net.eval()
    running_loss = []
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output = net(data)
            loss = F.l1_loss(output, target) + F.mse_loss(output, target)
            running_loss.append(loss.item())
    return np.mean(running_loss)


if __name__ == '__main__':
    log_dir = args.log_path
    ckp_dir = args.checkpoint_dir

    if args.restore_ckp:
        path_to_restore_checkpoint_file = args.restore_ckp

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((50, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = STNDataset(args.dataset_path, 'Train', 1.5, transform, transform)
    test_set = STNDataset(args.dataset_path, 'Test', 1.5, transform, transform)

    model = STN().to(DEVICE)
    last_epoch = -1

    train_size = int(args.train_fraction * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    # scheduler = StepLR(optimizer, step_size=args.decay_steps, gamma=0.9)
    # if args.multi_step:
    #     milestones = [int(v.strip()) for v in args.milestones.split(",")]
    #     scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    for epoch in range(last_epoch + 1, args.num_epochs):
        # scheduler.step()
        train(train_loader, model, optimizer, device=DEVICE, ep=epoch)

        if (epoch + 1) % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss = test(val_loader, model, DEVICE)
            logging.info(
                f"Epoch: {epoch + 1}, " +
                f"Validation Loss {np.mean(val_loss):.4f}"
            )
            path = model.store(ckp_dir, epoch + 1)
            logging.info(f"Saved model @'{path}'")
    print("\n--------------Finished Training!!----------------\n")
    print("\n--------------Evaluating on Test set!!-----------")
    test_loss = test(test_loader, model, DEVICE)
    logging.info(f"Validation Loss {test_loss:.4f}, ")
