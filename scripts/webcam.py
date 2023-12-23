from pathlib import Path

import cv2
import numpy as np
import torch
from localizer.model import Localizer

models_dir = Path(__file__).parent.parent / 'models'
model_file = models_dir / 'localizer.pth'

def main():
    vid = cv2.VideoCapture(0) 

    model = Localizer()
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f'Loaded weights from: {model_file}')

    model.eval()

    while(True): 
        ret, frame = vid.read() 
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break

        # Cropping image so it's square
        h, w, _ = frame.shape
        size = min(h, w)
        start_x, start_y = (w - size) // 2, (h - size) // 2
        frame = frame[start_y:start_y+size, start_x:start_x+size]

        # Performing inference
        input = cv2.resize(frame, (224, 224))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        input = torch.tensor(input, dtype=torch.float32)
        input = input.reshape((1, 1, 224, 224))
        input /= 255
        with torch.no_grad():
            _, corners = model.forward(input)
        corners = corners.squeeze().reshape((4, 2)) * size

        # Drawing result on frame
        pts = corners.reshape((4,1,2)).numpy().astype(np.int32)
        cv2.polylines(frame, [pts], True, color=(0,0,255), thickness=2)
    
        cv2.imshow('frame', frame) 
        if cv2.waitKey(1) == ord('q'): 
            break
    
    vid.release() 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()
