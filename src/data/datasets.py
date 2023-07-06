from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RandomImageDataset(Dataset):
    """Random images that do not contain sudokus."""

    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            transform_target (callable, optional): Optional transform to be applied
                on a target.
        """
        self.root_dir = Path(root_dir)
        self.images = [str(img_path) for img_path in self.root_dir.iterdir()]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path)
        # image = image.convert('RGB')

        min_size = min(image.width, image.height)
        if min_size < 224:
            scale = 224 / min_size
            image = image.resize((round(image.width * scale), round(image.height * scale)))

        #  No sudoku on image so it does not have a bounding box
        target = [0, 0, 0, 0]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        return image, target


class PuzzleDataset(Dataset):
    """Sudoku Puzzle dataset."""

    def __init__(self, csv_file, root_dir='.', transform=None, target_transform=None):
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
