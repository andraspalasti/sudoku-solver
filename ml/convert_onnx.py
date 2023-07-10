import argparse

import torch

from ml.models import Localizer, DigitClassifier

parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, choices=['localizer', 'digitclassifier'],
                    help='the model to export as onnx')


models = {
    'localizer': {
        'torch_model': Localizer,
        'checkpoint': 'models/localizer_best.pth.tar',
        'input_shape': (1, 1, 224, 224), 
        'input_names': ['input'],
        'output_names': ['classification', 'localization'],
        'out': 'localizer.onnx',
    },
    'digitclassifier': {
        'torch_model': DigitClassifier,
        'checkpoint': 'models/digitclassifier_best.pth.tar',
        'input_shape': (9*9, 1, 28, 28),
        'input_names': ['input'],
        'output_names': ['classification'],
        'out': 'digitclassifier.onnx',
    }
}

def main():
    args = parser.parse_args()
    model_def = models[args.model]
    torch_model = model_def['torch_model']()

    # Load model with weights
    checkpoint = torch.load(model_def['checkpoint'], map_location='cpu')
    torch_model.load_state_dict(checkpoint['state_dict'])

    # Set model to inference mode
    torch_model.eval()

    # Input to the model
    x = torch.randn(model_def['input_shape'], requires_grad=True)
    torch_out = torch_model(x)

    dynamic_axes = { name: [0] for name in model_def['output_names'] }
    dynamic_axes['input'] = [0]

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      model_def['out'],
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=14,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=model_def['input_names'],
                      output_names=model_def['output_names'],
                      dynamic_axes=dynamic_axes,
                      verbose=False)

    # To generate ort from onnx run:
    #  python3 -m onnxruntime.tools.convert_onnx_models_to_ort <onnx model file or dir>


if __name__ == '__main__':
    main()
