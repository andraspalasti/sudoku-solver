import time

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class Localizer(nn.Module):
    """Classifier and localizer for a sudoku puzzle, based on the MobileNetV3 architecture."""

    def __init__(self):
        super().__init__()

        # The classifier has one output which is a probability that 
        # indicates whether a sudoku is present or not
        self.mobilenet = mobilenet_v3_small(num_classes=1)

        last_out_channels = -1
        for m in self.mobilenet.features.modules():
            if isinstance(m, nn.Conv2d):
                last_out_channels = m.out_channels


        # The localization part has four outputs: x1, y1, x2, y2 
        # the corners of the bounding box
        self.localization = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(last_out_channels, 4),
            nn.ReLU(inplace=True)
        )

        for m in self.localization.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor):
        x = self.mobilenet.features(x)

        x = self.mobilenet.avgpool(x)
        x = torch.flatten(x, 1)

        classification = nn.functional.sigmoid(self.mobilenet.classifier(x))
        localization = self.localization(x)
        return classification, localization


if __name__ == '__main__':
    input = torch.randn((1, 3, 400, 400))

    model = Localizer()
    model.eval()

    num_params = sum(param.numel() for param in model.parameters())
    print(f'params: {num_params}')

    start = time.perf_counter_ns()
    is_present, location = model(input)
    end = time.perf_counter_ns()
    print(f'is_present: {is_present.shape} location: {location.shape}')
    print(f'{( end - start ) * 1e-6}ms')
