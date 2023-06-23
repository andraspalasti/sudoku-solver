from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FakeTarget():
    def __init__(self):
        pass

    def item(self):
        return torch.tensor([0, 1, 0, 0, 0, 0], dtype=torch.float32)


class PuzzleDataset(Dataset):
    """Sudoku Puzzle dataset."""

    def __init__(self, csv_file, root_dir = '.', transform=None, target_transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.puzzles_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.puzzles_frame)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #  Resize image to appropriate size
        img_path = self.root_dir / self.puzzles_frame.iloc[idx, 0]
        image = Image.open(str(img_path))
        scalex, scaley = 224 / image.width, 224 / image.height
        image = image.resize((224, 224))

        #  Adjust bounding box to resized image
        target = self.puzzles_frame.iloc[idx, 1:].values
        target = target.reshape((4, 2)) * [scalex, scaley]
        bbox = [target[:, 0].min(), target[:, 1].min(),
                target[:, 0].max(), target[:, 1].max()]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            bbox = self.target_transform(bbox)

        return image, bbox
