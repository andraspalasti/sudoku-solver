import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / 'data'

SUDOKUS_DIR = DATA_DIR / 'sudokus'
SUDOKU_REPO = 'https://raw.githubusercontent.com/wichtounet/sudoku_dataset/master'
SUDOKU_FILES = [
    'datasets/v2_train.tar.bz2',
    'datasets/v2_test.tar.bz2',
    'outlines_sorted.csv',
]


def download_sudoku_dataset():
    SUDOKUS_DIR.mkdir(parents=True, exist_ok=True)
    for file in SUDOKU_FILES:
        fname = file.split('/')[-1]
        download(f'{SUDOKU_REPO}/{file}', SUDOKUS_DIR / fname)

    compressed = list(DATA_DIR.glob('**/*.tar.bz2'))
    if 0 < len(compressed):
        for f in compressed:
            print(f'Extracting: {f}')
            tarfile.open(f).extractall(f.parent)


def download(url: str, fpath, chunk_size=1024):
    fpath = Path(fpath)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fpath, 'wb') as file, tqdm(
        desc=fpath.name,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


if __name__ == '__main__':
    download_sudoku_dataset()

