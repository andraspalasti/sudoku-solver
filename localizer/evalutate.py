import torch
from torch import Tensor
from torchvision.ops import box_iou
from tqdm import tqdm


def corners_to_box(corners: Tensor) -> Tensor:
    corners = corners.reshape((-1, 4, 2))
    min_xy = corners.min(dim=1).values
    max_xy = corners.max(dim=1).values
    return torch.cat((min_xy, max_xy), dim=1).squeeze()


@torch.inference_mode()
def evaluate(net, dataloader, device: torch.device):
    net.eval()
    num_batches = len(dataloader)

    iou_score = 0.0
    for batch in tqdm(dataloader, total=num_batches, desc='Evaluation', unit='batch', leave=False):
        images, corners = batch

        images = images.to(device=device)
        corners = corners.to(device=device)

        # TODO: Currently we don't compare object detection error
        _, corner_preds = net(images)
        bbox, bbox_preds = corners_to_box(corners), corners_to_box(corner_preds)

        _eye = torch.eye(corners.shape[0], device=device, dtype=torch.bool)
        iou_score += box_iou(bbox, bbox_preds)[_eye].mean()
    return iou_score / max(num_batches, 1)
