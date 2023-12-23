from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import mobilenet_v2


class Localizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # MobileNet V2 pretrained backbone
        mobilenet = mobilenet_v2(
            weights='MobileNet_V2_Weights.DEFAULT'
        )

        self.expand = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.__freezed = False
        self.backbone = mobilenet.features
        self.last_channel = mobilenet.last_channel

        self.localizer = nn.Linear(self.last_channel, 1+4*2)

        # weights = list(self.localizer.weight[1:].split(split_size=1))
        # for i, (xW, yW) in enumerate(zip(weights[::2], weights[1::2])):
            # print(i, xW.shape, yW.shape)
        # self.localizer.weight.data += 0.001
        # nn.init.kaiming_normal_(self.localizer.weight, nonlinearity='relu')

    def freeze(self):
        if self.__freezed: return
        self.__freezed = True
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        if not self.__freezed: return
        self.__freezed = False
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.expand(x)
        x = self.backbone(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, start_dim=1)
        x = self.localizer(x)
        return F.sigmoid(x[:, 0]), x[:, 1:]

if __name__ == '__main__':
    from pathlib import Path

    from localizer.dataset import SudokuDataset

    # Fix for reproducibility
    torch.manual_seed(42842)

    sudokus_dir = Path(__file__).parent.parent / 'data' / 'sudokus'
    train_set = SudokuDataset(sudokus_dir / 'v2_train', sudokus_dir / 'outlines_sorted.csv')

    img, _ = train_set[0]
    img = img.reshape((1, 1, 224, 224))
    # img = (img - 0.5367787160068429) / 0.12944043373444886

    model = Localizer()
    model.eval()

    with torch.no_grad():
        pred = model(img) 
        print(pred)
