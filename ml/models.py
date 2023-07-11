import time

import torch
import torch.nn as nn
from torchvision.models import MobileNetV2


class Localizer(nn.Module):
    """Classifier and localizer for a sudoku puzzle, based on the MobileNetV2 architecture."""

    def __init__(self):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        # The classifier has one output which is a probability that 
        # indicates whether a sudoku is present or not
        self.mobilenet = MobileNetV2(num_classes=1)
        last_out_channels = [m.out_channels for m in self.mobilenet.modules()
                             if isinstance(m, nn.Conv2d)][-1]

        # The localization part has four outputs: x1, y1, x2, y2 
        # the corners of the bounding box
        self.localization = nn.Sequential(
            nn.Linear(last_out_channels, 4),
            nn.ReLU(inplace=True)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor):
        x = self.pre(x)
        x = self.mobilenet.features(x)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, start_dim=1)

        classification = nn.functional.sigmoid(self.mobilenet.classifier(x))
        localization = self.localization(x)
        return classification, localization


class DigitClassifier(nn.Module):
    """Classifier for handwritten digits."""

    def __init__(self, dropout=0.5):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), # 28x28 input
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 14x14 output

            nn.Conv2d(8, 32, kernel_size=3), # input: 14x14, output: 12x12
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # input: 12x12, output 6x6

            nn.Conv2d(32, 128, kernel_size=3, padding=1), # 6x6 input
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2), # 3x3 output

            nn.Conv2d(128, 512, kernel_size=3), # input: 3x3, output: 1x1
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        n_hidden = 256
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, 10)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == '__main__':
    input = torch.randn((1, 1, 224, 224))
    model = Localizer()
    model.eval()

    num_params = sum(param.numel() for param in model.parameters())
    print('Localizer: ')
    print(f'params: {num_params}')
    with torch.no_grad():
        start = time.perf_counter_ns()
        is_present, location = model(input)
        end = time.perf_counter_ns()
    print(f'is_present: {is_present.shape} location: {location.shape}')
    print(f'{( end - start ) * 1e-6}ms\n')


    input = torch.randn((1, 1, 28, 28))
    model = DigitClassifier()
    model.eval()

    num_params = sum(param.numel() for param in model.parameters())
    print('DigitClassifier: ')
    print(f'params: {num_params}')
    with torch.no_grad():
        start = time.perf_counter_ns()
        classification = model(input)
        end = time.perf_counter_ns()
    print(f'classification: {classification.shape}')
    print(f'{( end - start ) * 1e-6}ms')
