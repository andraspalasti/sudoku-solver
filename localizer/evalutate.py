from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from torch import Tensor
from tqdm import tqdm


def corners_to_box(corners: Tensor) -> Tensor:
    corners = corners.reshape((-1, 4, 2))
    min_xy = corners.min(dim=1).values
    max_xy = corners.max(dim=1).values
    return torch.cat((min_xy, max_xy), dim=1).squeeze()

def iou_polygon(corners: Tensor, corner_preds: Tensor) -> float:
    p1 = Polygon(corners.reshape((4, 2)).cpu().tolist())
    p2 = Polygon(corner_preds.reshape((4, 2)).cpu().tolist())
    intersect = p1.intersection(p2).area
    return intersect / (p1.area + p2.area - intersect)


def corners_to_mask(corners: Tensor, size: Tuple[int, int]):
    corners = (corners.reshape((4, 2)) * 224).cpu().tolist()
    corners = [tuple(xy) for xy in corners]
    mask = Image.new(mode='L', size=size, color=0)
    ImageDraw.Draw(mask).polygon(corners, fill=1, outline=1)
    return np.asarray(mask)


@torch.inference_mode()
def evaluate(net, dataloader, device: torch.device):
    net.eval()

    iou_score, num_samples = 0.0, 0
    for batch in tqdm(dataloader, total=len(dataloader), desc='Evaluation', unit='batch', leave=False):
        images, corners = batch

        images = images.to(device=device)
        corners = corners.to(device=device)

        # TODO: Currently we don't compare object detection error
        _, corner_preds = net(images)

        for shapes in zip(torch.split(corners, 1), torch.split(corner_preds, 1)):
            try:
                iou_score += iou_polygon(*shapes)
            except GEOSException: # Wrong topology of shape
                iou_score += 0

        num_samples += corners.shape[0]

    return iou_score / max(num_samples, 1)

