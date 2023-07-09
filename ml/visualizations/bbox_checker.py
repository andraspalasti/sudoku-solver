from pathlib import Path

import cv2 
import pandas as pd


data_dir = Path('data')

def main():
    df = pd.read_csv(str(data_dir / 'outlines_sorted.csv'))
    df = df[df['filepath'].str.contains('image')]

    cv2.namedWindow('preview')

    for i in range(len(df)):
        img_path, bpoints = df.iloc[i, 0], df.iloc[i, 1:]
        x1, y1 = int(bpoints.values[::2].min()), int(bpoints.values[1::2].min())
        x2, y2 = int(bpoints.values[::2].max()), int(bpoints.values[1::2].max())

        img = cv2.imread(str(data_dir / img_path))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)
        cv2.imshow('preview', img)
        print(f'Current: {i} {img_path}')

        key_code = cv2.waitKey()
        if key_code == 27: # Exit on ESC
            break

    cv2.destroyWindow('preview')

if __name__ == '__main__':
    main()
