import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import albumentations as A

transform = A.Compose([
    A.OneOf([
        A.Sharpen(),
        A.GaussNoise(),
        A.GaussianBlur(),
    ]),
    A.OneOf([
        A.MotionBlur(),
        A.MedianBlur(blur_limit=1),
        A.Blur(blur_limit=1),
    ]),
    A.Compose([
        A.CropAndPad(px=25, pad_mode=1, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=(-0.5, 0), rotate_limit=10,
                           border_mode=cv2.BORDER_CONSTANT, always_apply=True),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),            
    ]),
    A.HueSaturationValue(),
], keypoint_params=A.KeypointParams(format='xy'))


sudokus_dir = Path(__file__).parent.parent / 'data' / 'sudokus'
config = {
    'train': (
        sudokus_dir / 'v2_train',
        sudokus_dir / 'outlines_sorted.csv'
    ),
    'test': (
        sudokus_dir / 'v2_test',
        sudokus_dir / 'outlines_sorted.csv'
    )
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passes', type=int, default=1, help='Number of times augmentation should be performed')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='On which split should we perform augmentation')
    args = parser.parse_args()

    imgs_dir, csv_path = config[args.split]
    images = list(imgs_dir.glob('image*.jpg'))
    original = pd.read_csv(csv_path, sep=',')

    # Preprocess csv
    locs = original.copy()
    locs['image_id'] = locs['filepath'].str.split('/').str[-1].str.rstrip('.jpg')
    locs = locs[locs['image_id'].str.startswith('image')]
    locs = locs.set_index('image_id', drop=True)
    locs = locs.drop(columns=['filepath'])

    id = 0
    augmented = []
    for _ in range(args.passes):
        for file in (t := tqdm(images, total=len(images), desc=f'Augmenting images', unit='img')):
            image_id = file.name.split('.jpg')[0]
            if image_id not in locs.index: 
                t.set_postfix_str(f'Skipped (not found): {image_id}')
                continue

            corners = locs.loc[image_id].values.reshape((4, 2))
            img = Image.open(file)
            img = ImageOps.exif_transpose(img)
            img = np.asarray(img)

            # TODO: Maybe introduce 90 degree rotations
            transformed = transform(image=img, keypoints=corners.tolist())
            if np.all(img == transformed['image']):
                t.set_postfix_str(f'Skipped (unchanged): {image_id}')
                continue

            fname = f'aug_{image_id}_{id}.jpg'
            img = transformed['image']
            corners = np.array(transformed['keypoints'])
            if not corners.shape == (4, 2):
                t.set_postfix_str(f'Skipped (keypoint out): {image_id}')
                continue

            augmented.append([fname] + corners.flatten().tolist())
            Image.fromarray(img).save(imgs_dir / fname)
            id += 1

    original = pd.concat([
        pd.DataFrame(augmented, columns=original.columns),
        original
    ])
    original = original.drop_duplicates(subset='filepath')
    original.to_csv(csv_path, sep=',', index=False)


if __name__ == '__main__':
    random.seed(42) 
    main()
