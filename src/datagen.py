from typing import Literal

import pandas as pd

from sklearn.datasets import make_moons, make_circles, make_blobs
import matplotlib.pyplot as plt

from src.utils import PGException


def get_dataset(random_state, n_samples, generate_type='moons'):
    x, y = None, None
    if generate_type == 'moons':
        x, y = make_moons(noise=0.09, random_state=random_state, n_samples=n_samples)
    if generate_type == 'circles':
        x, y = make_circles(noise=0.09, random_state=random_state, n_samples=n_samples, factor=0.5)
    if generate_type == 'blobs':
        x, y = make_blobs(random_state=random_state, n_samples=n_samples, centers=2)
    return x, y


def draw_dataset(x, y):
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()


def datagen(name: str, data_type: Literal["moons", "circles", "blobs"], n_samples: int, random_seed: int):
    x, y = get_dataset(random_seed, n_samples, data_type)
    if x is None or y is None:
        raise PGException('Unknown data type')

    data = pd.DataFrame()
    data['X1'] = x[:, 0]
    data['X2'] = x[:, 1]
    data['y'] = y
    data.to_csv(name)
