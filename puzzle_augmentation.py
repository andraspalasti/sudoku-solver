import numpy as np
import albumentations as A
import cv2
import pandas as pd
from pathlib import Path

puzzles_dir = Path('puzzles')
df = pd.read_csv('outlines_sorted.csv')
df = df[df['filepath'].str.contains('image')]

augmentations = [
    # (A.Compose([
    # A.Rotate(limit=10),
    # ], bbox_params=A.BboxParams('pascal_voc')), 10),
    (A.Compose([
        A.Affine(scale=0.6, translate_percent=(-0.2, 0.2), rotate=0, shear=0),
    ], bbox_params=A.BboxParams('pascal_voc')), 400),
    (A.Compose([
        A.GaussianBlur(),
    ], bbox_params=A.BboxParams('pascal_voc')), 100),
]


def apply(ix: int, transform: A.Compose):
    img_name = df.iloc[ix, 0]
    bbox = df.iloc[ix, 1:]
    assert len(bbox) == 4 * 2  #  4 points every point has two coordinates

    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(str(puzzles_dir / img_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    min_x, min_y = min(bbox[::2]), min(bbox[1::2])
    max_x, max_y = max(bbox[::2]), max(bbox[1::2])

    transformed = transform(image=image, bboxes=[
                            [min_x, min_y, max_x, max_y, 'sudoku']])
    return transformed['image'], transformed['bboxes'][0]


if __name__ == '__main__':
    new_rows = []

    for transform, count in augmentations:
        ixs = np.random.choice(len(df), count)
        for ix in ixs:
            img, bbox = apply(ix, transform)
            min_x, min_y, max_x, max_y, _ = bbox
            new_rows.append([img,
                            min_x, min_y,
                            max_x, min_y,
                            min_x, max_y,
                            max_x, max_y])

    offset = len(df)
    for i, new_row in enumerate(new_rows):
        img_name = f'transformed{i}.jpg'
        cv2.imwrite(str(puzzles_dir / img_name), img=new_row[0])
        new_row[0] = img_name
        df.loc[offset+i] = new_row

    df.to_csv('outlines_sorted.csv', index=False)
