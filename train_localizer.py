import os
import torch
import pandas as pd
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision.datasets import FakeData
from torchvision.ops import generalized_box_iou_loss
from PIL import Image
from models import Localizer


class FakeLabel():
    def __init__(self):
        pass

    def item(self):
        return torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)


class PuzzleDataset(Dataset):
    """Sudoku Puzzle dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.puzzles_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.puzzles_frame)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir,
                                self.puzzles_frame.iloc[idx, 0])
        image = Image.open(img_path)
        scalex, scaley = 224 / image.width, 224 / image.height
        image = image.resize((224, 224))
        if self.transform:
            image = self.transform(image)

        # Because we resized the image we need to resize the bounding points
        target = self.puzzles_frame.iloc[idx, 1:].values
        target = target.reshape((4, 2)) * [scalex, scaley]
        bbox = [target[:,0].min(), target[:,1].min(), target[:,0].max(), target[:,1].max()]
        if self.target_transform:
            bbox = self.target_transform(bbox)

        return image, bbox


def loss_fn(outputs, labels):
    loss = F.binary_cross_entropy(outputs[0], labels[:,:2])
    loss += F.mse_loss(outputs[1], labels[:, 2:])
    return loss

def train_one_epoch(model, training_loader: DataLoader, optimizer: torch.optim.Optimizer):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to('mps'), labels.to('mps')

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5  # loss per batch
            print(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0.

    return last_loss


def main():
    model = Localizer()
    model.to('mps')

    dataset = ConcatDataset([
        FakeData(
          transform=transforms.ToTensor(),
          target_transform=lambda _: FakeLabel()
        ),
        PuzzleDataset(
            'outlines_sorted.csv', 'puzzles',
            transform=transforms.ToTensor(),
            target_transform=lambda bbox: torch.tensor([1, 0, *bbox], dtype=torch.float32),
        ),
    ])

    train_size = int(len(dataset) * 0.9)
    validation_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])

    training_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=32, shuffle=False)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch+1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, training_loader, optimizer)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to('mps'), vlabels.to('mps')
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
            # best_vloss = avg_vloss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # torch.save(model.state_dict(), model_path)


if __name__ == '__main__':
    main()
