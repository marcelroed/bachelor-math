import zipfile
from pathlib import Path

from tensorflow.keras.datasets import mnist
import numpy as np
import requests

MNIST_ROT_URL = "https://www.dropbox.com/s/0fxwai3h84dczh0/mnist_rotation_new.zip?dl=1"
DATA_DIR = Path(__file__).parents[2] / 'data'


def get_mnist_rot(with_valid=False):
    folder = DATA_DIR / 'mnist_rot'
    if not folder.is_dir():
        # Download to data directory
        stream = requests.get(MNIST_ROT_URL)

        folder.mkdir()
        zip_file = folder / 'mnist_rot.zip'
        with open(zip_file, 'wb') as f:
            f.write(stream.content)
        if not zipfile.is_zipfile(zip_file):
            folder.rmdir()
            raise ValueError(f'Downloaded file is not a zip-file! Maybe the URL {MNIST_ROT_URL} is broken.')

        archive = zipfile.ZipFile(str(zip_file), mode='r')
        archive.extractall(str(folder))
        archive.close()
        for file in (folder / 'mnist_rotation_new').iterdir():
            file.replace(folder / file.name)
        (folder / 'mnist_rotation_new').rmdir()
        zip_file.unlink()
        print('Successfully downloaded and extracted rotated MNIST dataset')

    test, train, valid = [(ds['x'], ds['y'])
                          for ds in [np.load(str(folder / f'rotated_{s}.npz')) for s in ('test', 'train', 'valid')]]
    if not with_valid:
        train = (np.vstack((train[0], valid[0])), np.hstack((train[1], valid[1])))
        return train, test

    return train, test, valid


def to_one_hot(v, no_classes):
    labels = np.zeros((v.shape[0], no_classes), float)
    labels[np.arange(v.shape[0]), v] = 1

    return labels


def get_mnist():
    dataset = mnist.load_data()
    no_classes = 10

    train, test = map(lambda t: ((t[0] if len(t[0].shape) == 4 else t[0][..., np.newaxis]).astype(float) / 255.0,
                                 to_one_hot(t[1] if len(t[1].shape) == 1 else t[1][:, 0], no_classes)), dataset)

    print(train[0].shape)
    return train, test


if __name__ == '__main__':
    get_mnist_rot()
