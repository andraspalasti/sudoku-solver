import argparse
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='outlines_sorted.csv',
                    help='path to csv that contains the images to be augmented')
parser.add_argument('-o', '--out-dir', default='puzzles',
                    help='output directory to put transformed images in')
parser.add_argument('--nrot', type=int, default=1,
                    help='number of rotations to perform')
parser.add_argument('--ntrans', type=int, default=1,
                    help='number of translations to perform')
parser.add_argument('--nblur', type=int, default=1,
                    help='number of gaussian blurs to perform')


def apply(img_path: str, bpoints: list, transform: A.Compose):
    assert len(bpoints) == 4 * 2  #  4 points every point has two coordinates

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    min_x, min_y = min(bpoints[::2]), min(bpoints[1::2])
    max_x, max_y = max(bpoints[::2]), max(bpoints[1::2])

    transformed = transform(image=image, bboxes=[
                            [min_x, min_y, max_x, max_y, 'sudoku']])
    return transformed['image'], transformed['bboxes'][0]


def main():
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df['filepath'].str.contains('image')]

    augmentations = [
        (A.Compose([
            A.Affine(scale=0.8, translate_percent=0.0, rotate=15.0, shear=0),
        ], bbox_params=A.BboxParams('pascal_voc')), args.nrot),
        (A.Compose([
            A.Affine(scale=0.6, translate_percent=(-0.2, 0.2), rotate=0, shear=0),
        ], bbox_params=A.BboxParams('pascal_voc')), args.ntrans),
        (A.Compose([
            A.GaussianBlur(),
        ], bbox_params=A.BboxParams('pascal_voc')), args.nblur),
    ]

    new_rows = []

    for transform, count in augmentations:
        ixs = np.random.choice(len(df), count)
        for ix in ixs:
            img_path = df.iloc[ix, 0]
            bpoints = df.iloc[ix, 1:]  #  Bounding points of the sudoku

            img, bbox = apply(img_path, bpoints, transform)
            min_x, min_y, max_x, max_y, _ = bbox
            new_rows.append([img,
                            min_x, min_y,
                            max_x, min_y,
                            min_x, max_y,
                            max_x, max_y])

    offset = len(df)
    for i, new_row in enumerate(new_rows):
        img_path = os.path.join(args.out_dir, f'transformed{i}.jpg')
        cv2.imwrite(img_path, img=new_row[0])
        new_row[0] = img_path
        df.loc[offset+i] = new_row

    df.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
