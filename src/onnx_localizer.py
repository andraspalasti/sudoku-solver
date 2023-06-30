import torch
import torchvision.transforms.functional as TF

from models import Localizer


class ONNXLocalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Localizer()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(dim=0)
        classification, localization = self.net(x)
        return classification[0], localization[0]


if __name__ == '__main__':
    torch_model = ONNXLocalizer()

    # Load model with weights
    checkpoint = torch.load('models/model_bestV3.pth.tar', map_location='cpu')
    torch_model.net.load_state_dict(checkpoint['state_dict'])

    # Set model to inference mode
    torch_model.eval()

    # Input to the model
    x = torch.randn(1, 400, 400, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "localizer.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['classification', 'localization'])  # the model's output names

    # To generate ort from onnx run:
    #  python3 -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>
