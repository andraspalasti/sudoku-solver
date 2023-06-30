import cv2
import torch

from models import Localizer


def main():
    device = torch.device('cpu')

    model = Localizer().to(device)
    model.eval()

    # Load weights
    checkpoint = torch.load('models/model_bestV3.pth.tar', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        topx, topy = 0, 0
        topy, topx = (image.shape[0] - 500) // 2, (image.shape[1] - 500) // 2
        image = image[topy:topy+500, topx:topx+500]
        scaley, scalex = image.shape[0] / 400, image.shape[1] / 400 
        image = cv2.resize(image, (400, 400))

        input = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(dim=0)
        # input = torch.cat((input, input, input))
        input = input / 255

        with torch.no_grad():
            probs, bbox = model(input.unsqueeze(dim=0))

        bbox[0, ::2] = bbox[0, ::2] * scalex 
        bbox[0, 1::2] = bbox[0, 1::2] * scaley
        bbox = bbox[0].tolist()
        present_prob = round(probs[0, 0].item() * 100)
        x1, y1 = int(topx + bbox[0]), int(topy + bbox[1])
        x2, y2 = int(topx + bbox[2]), int(topy + bbox[3])

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)
        frame = cv2.rectangle(frame, (topx, topy), (topx+500, topy+500), color=(0, 255, 0), thickness=3)
        frame = cv2.putText(frame, f'{present_prob}%', org=(x1+10, y1+20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 0, 255))

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyWindow("preview")

if __name__ == '__main__':
    main()
