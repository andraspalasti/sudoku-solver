import argparse
import os
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from data.datasets import RandomImageDataset, PuzzleDataset, MyMNIST
from models import Localizer, DigitClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cpu',
                    help='the device that should perform the computations')
parser.add_argument('--model', default='digitclassifier', choices=['localizer', 'digitclassifier'],
                    help='the model that should be trained')
parser.add_argument('--epochs', default=50, type=int, metavar='N', choices=range(10000),
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int, choices=[2 ** x for x in range(1, 9)],
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='learning_rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def train(training_loader, model, criterion, optimizer, epoch, device):
    #  Switch to training mode
    model.train()

    progress = tqdm(training_loader, desc=f'EPOCH {epoch}')

    running_loss = 0.0
    for i, (images, targets) in enumerate(progress):

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix({ 'AvgLoss': running_loss / (i+1) })

    return running_loss / (i+1)


def validate(val_loader, model, criterion, device):
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                # move data to the same device as model
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                voutputs = model(inputs)
                vloss = criterion(voutputs, targets)
                running_vloss += vloss
        return running_vloss / (i+1)


def get_localizer(device):
    model = Localizer()
    model.to(device)

    dataset = ConcatDataset([
        *([RandomImageDataset(
            'data/images',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
            ]),
            target_transform=lambda bbox: torch.tensor(
                [0, *bbox], dtype=torch.float32),
        )] * 10),
        PuzzleDataset(
            csv_file='data/outlines_sorted.csv',
            root_dir='data',
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
            target_transform=lambda bbox: torch.tensor(
                [1, *bbox], dtype=torch.float32),
        ),
    ])
    train_size = int(len(dataset) * 0.8)
    validation_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])

    #  Loss function
    def criterion(outputs, targets: torch.Tensor):
        # the classification has a big impact too
        loss = F.binary_cross_entropy(outputs[0], targets[:, :1]) * 1000

        # images that does not have sudokus in it, should not have impact on the localizer
        # so we only compute loss for the images that contain sudokus
        present = torch.tensor([1], dtype=torch.float32, device=device)
        ixs = torch.all(targets[:, :1] == present, dim=1)
        loss += F.mse_loss(outputs[1][ixs], targets[ixs, 1:])
        return loss

    return model, train_set, validation_set, criterion


def get_digit_classifier(device):
    model = DigitClassifier()
    model.to(device)

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=15, translate=(0.3, 0.3))
    ])
    train_set = MyMNIST('data', train=True, transform=t)
    validation_set = MyMNIST('data', train=False, transform=t)

    #  Loss function
    def criterion(logits: torch.Tensor, targets: torch.Tensor):
        return F.nll_loss(logits, targets)

    return model, train_set, validation_set, criterion

def main():
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.model == 'localizer':
        model, train_set, validation_set, criterion = get_localizer(device)
    else:
        model, train_set, validation_set, criterion = get_digit_classifier(device)

    training_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # Sets the learning rate to the initial LR decayed by 10 every 300 epochs
    scheduler = StepLR(optimizer, step_size=300, gamma=0.1)
    best_loss = 1 << 32

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)

            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at: '{args.resume}'")

    if args.evaluate:
        avg_vloss = validate(validation_loader, model, criterion, device)
        print(f'avarage validation loss: {avg_vloss}')
        return

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train(training_loader, model, criterion, optimizer, epoch, device)

        avg_vloss = validate(validation_loader, model, criterion, device)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        is_best = best_loss > avg_vloss
        best_loss = max(best_loss, avg_vloss)

        scheduler.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


if __name__ == '__main__':
    main()
