import time
from itertools import chain

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNetV2


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        out_dim = 6
        self.classifier = MobileNetV2(num_classes=out_dim)

        self.is_present = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, 2),
            nn.Softmax(dim=1)
        )

        self.location = nn.Sequential(
            nn.ReLU(),
            nn.Linear(out_dim, 4),
            nn.ReLU(inplace=True)
        )

        modules = chain(self.is_present.modules(), self.location.modules())
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        out = self.classifier.forward(x)
        is_present = self.is_present(out)
        location = self.location(out)
        # is_present = torch.nn.functional.softmax(out[:, :2], dim=1)
        # location = torch.nn.functional.relu(out[:, 2:])
        return is_present, location


if __name__ == '__main__':
    input = torch.randn((1, 3, 224, 224)).to("cpu")

    model = Localizer()
    model.to("cpu")
    model.eval()

    num_params = sum(param.numel() for param in model.parameters())
    print(f'params: {num_params}')

    start = time.perf_counter_ns()
    is_present, location = model(input)
    end = time.perf_counter_ns()
    print(f'is_present: {is_present.shape} location: {location.shape}')
    print(f'{( end - start ) * 1e-6}ms')
