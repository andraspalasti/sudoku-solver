import torch
import torchvision.transforms.functional as TF

from models import Localizer


if __name__ == '__main__':
    torch_model = Localizer()

    # Load model with weights
    checkpoint = torch.load('models/model_best.pth.tar', map_location='cpu')
    torch_model.load_state_dict(checkpoint['state_dict'])

    # Set model to inference mode
    torch_model.eval()

    # Input to the model
    x = torch.randn((1, 1, 224, 224), requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      "localizer.onnx",
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=14,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['classification', 'localization'])  # the model's output names

    # To generate ort from onnx run:
    #  python3 -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>
