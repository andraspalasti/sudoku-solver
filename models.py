import time
import torch
import torch.nn as nn
from torchvision.models import MobileNetV2


class Localizer(nn.Module):
    def __init__(self):
        super().__init__()
        out_dim = 1000
        self.classifier = MobileNetV2(num_classes=out_dim)
        self.is_present = nn.Sequential(
            nn.Linear(out_dim, 2, bias=False),
            nn.Softmax(dim=1)
        )
        self.location = nn.Sequential(
            nn.Linear(out_dim, 4, bias=False),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        out = self.classifier.forward(x)
        is_present = self.is_present(out)
        location = self.location(out)
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
