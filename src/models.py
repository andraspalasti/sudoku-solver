import time
from itertools import chain

import torch
import torch.nn as nn
from torchvision.models import MobileNetV2


class Localizer(MobileNetV2):
    """Classifier and localizer for a sudoku puzzle, based on the MobileNetV2 architecture."""
    def __init__(self):
        # The classifier has one output which is a probability that 
        # indicates whether a sudoku is present or not
        super().__init__(num_classes=1)

        # The localization part has four outputs: x1, y1, x2, y2 
        # the corners of the bounding box
        self.localization = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(self.last_channel, 4),
            nn.ReLU(inplace=True)
        )

        for m in self.localization.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        classification = nn.functional.sigmoid(self.classifier(x))
        localization = self.localization(x)

        return classification, localization


if __name__ == '__main__':
    input = torch.randn((1, 3, 224, 224))

    model = Localizer()
    model.eval()

    num_params = sum(param.numel() for param in model.parameters())
    print(f'params: {num_params}')

    start = time.perf_counter_ns()
    is_present, location = model(input)
    end = time.perf_counter_ns()
    print(f'is_present: {is_present.shape} location: {location.shape}')
    print(f'{( end - start ) * 1e-6}ms')
