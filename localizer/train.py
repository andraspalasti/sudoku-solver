import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from localizer.dataset import SudokuDataset
from localizer.evalutate import corners_to_box, evaluate
from localizer.model import Localizer

SUDOKUS_DIR = Path(__file__).parent.parent / 'data' / 'sudokus'
CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints'


def criterion(pred: Tuple[Tensor, Tensor], target: Tuple[Tensor, Tensor]):
    class_loss = F.binary_cross_entropy(pred[0], target[0])
    localization_loss = F.mse_loss(pred[1], target[1])
    return localization_loss
    return class_loss + localization_loss


def transforms(img: Tensor, corners: Tensor):
    return img.unsqueeze(0), corners


def train_model(
    model: Localizer,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float
):
    # Create datasets
    localizations = SUDOKUS_DIR / 'outlines_sorted.csv'
    train_set = SudokuDataset(SUDOKUS_DIR / 'v2_train', localizations, transforms)
    validation_set = SudokuDataset(SUDOKUS_DIR / 'v2_test', localizations, transforms)
    n_train, n_test = len(train_set), len(validation_set)

    # Creat data loaders
    loader_args = dict(batch_size=batch_size, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(validation_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize wandb)
    experiment = wandb.init(project='Sudoku-Localizer', resume='allow', anonymous='must')
    experiment.config.update({
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
    })

    print(f'''Starting training:
        Epochs:        {epochs}
        Batch size:    {batch_size}
        Learning rate: {learning_rate}
        Training size: {n_train}
        Test size:     {n_test}
    ''')

    # TODO: Resaearch optimizers
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # Begin training
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, corners = batch
                images = images.to(device=device)
                corners = corners.to(device=device)

                preds = model(images)
                loss = criterion(
                    preds,
                    (torch.ones(size=(corners.shape[0],), device=device), corners)
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        val_score = evaluate(model, val_loader, device)
        scheduler.step(val_score)
        min_x, min_y, max_x, max_y = corners_to_box(preds[1][0]).cpu().tolist()
        experiment.log({
            'learning rate': optimizer.param_groups[0]['lr'],
            'validation iou': val_score,
            'images': wandb.Image(images[0].cpu(), boxes={
                'predictions': {
                    'box_data': [{
                        'position': {'minX': min_x, 'maxX': max_x, 'minY': min_y, 'maxY': max_y},
                        'box_caption': 'sudoku',
                        'class_id': 0,
                    }]
                }
            }),
            'step': global_step,
            'epoch': epoch,
        })

        if epoch % 5 == 0:
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(CHECKPOINT_DIR / f'localizer_epoch{epoch}.pth'))
            print(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    # Fix for reproducibility
    torch.manual_seed(42842)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    model = Localizer()
    model.to(device)

    try:
        train_model(
            model=model,
            epochs=40,
            batch_size=5,
            learning_rate=0.001,
            device=device,
        )
    except torch.cuda.OutOfMemoryError:
        print('Detected OutOfMemoryError!')
        torch.cuda.empty_cache()
