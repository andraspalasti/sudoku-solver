from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


class SudokuDataset(Dataset):
    def __init__(self, images_dir, bboxes, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        self.images_dir = Path(images_dir)

        locs = pd.read_csv(bboxes, sep=',')
        locs['image_id'] = locs['filepath'].str.split('/').str[-1].str.rstrip('.jpg')

        image_ids = set(f.name.rstrip('.jpg') for f in self.images_dir.glob('*.jpg'))
        locs = locs[locs['image_id'].isin(image_ids)]
        locs = locs.set_index('image_id', drop=True)
        locs = locs.drop(columns=['filepath'])
        self.locs = locs

    def _get_image(self, image_id):
        img_path = self.images_dir / f'{image_id}.jpg'
        img = Image.open(img_path)
        ImageOps.exif_transpose(img, in_place=True)
        return img

    def _get_corners(self, image_id):
        corners = np.array(self.locs.loc[image_id]).reshape((4, 2))
        return corners

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_id = self.locs.iloc[[index]].index[0]
        img = self._get_image(image_id)
        corners = self._get_corners(image_id)
        img, corners = self.preprocess(img, corners)
        if self.transforms is not None:
            return self.transforms(img, corners)
        return img, corners

    @staticmethod
    def preprocess(img: Image.Image, bbox: Optional[np.ndarray] = None):
        scalex, scaley = 224 / img.width, 224 / img.height
        img = img.convert('L')
        img = img.resize((224, 224))
        img: Tensor = torch.tensor(img.getdata(), dtype=torch.float32)
        img = img.reshape((224, 224)) / 255
        if bbox is not None:
            bbox = bbox.reshape((-1, 2)) * [scalex, scaley] / 224
            bbox: Tensor = torch.from_numpy(bbox.reshape(-1))
        return img, bbox.type(torch.float32)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    sudokus_dir = Path(__file__).parent.parent / 'data' / 'sudokus'
    train_set = SudokuDataset(sudokus_dir / 'v2_train', sudokus_dir / 'outlines_sorted.csv')

    img, corners = train_set[0]
    corners = torch.cat((corners, corners[:2]))
    corners *= 224

    plt.imshow(img.numpy(), cmap='gray')
    plt.plot(corners[::2], corners[1::2], color='r', linewidth=2)
    plt.show()
