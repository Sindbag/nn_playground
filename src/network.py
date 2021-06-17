from typing import Dict, Literal

import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size: int, layers: Dict[Literal["size", "activation"], str], output: int):
        super().__init__()
        self.layers = []

        sizes = [input_size] + [layer['size'] for layer in layers]
        for i in range(1, len(sizes)):
            self.layers.append(nn.Linear(sizes[i - 1], sizes[i]))
            self.layers.append(self.get_activation(layers[i - 1]["activation"]))
        self.layers.append(nn.Linear(layers[-1]["size"], output))

        self.runner = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.runner(x)

    @staticmethod
    def get_activation(func_name: str):
        available_funcs = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigm': nn.Sigmoid(),
        }
        return available_funcs.get(func_name.lower(), nn.ReLU)
