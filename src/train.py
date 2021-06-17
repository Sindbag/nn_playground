import logging

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import yaml

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .dataloader import NumpyDataset
from .network import Network
from .plot import make_grid, plot_predictions


class Trainer:
    def __init__(self, num_epochs, learnig_rate, model, criterion, optimizer):
        self.num_epochs = num_epochs
        self.learnig_rate = learnig_rate
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, dataset: str, train_dataloader):
        name = dataset.split('.')[0]
        for epoch in range(self.num_epochs):
            for i, (features, labels) in enumerate(train_dataloader):
                y_predicted = self.model(features)
                loss = self.criterion(y_predicted, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch:
                    train_dataset = train_dataloader.dataset
                    x_train, x_test, y_train, y_test = get_data_from_datasets(train_dataset, train_dataset)

                    xx, yy = make_grid(x_train, x_test, y_train, y_test)
                    Z = predict_proba_on_mesh_tensor(self, xx, yy)

                    title = "frames_{}/frame_{}.png".format(name, epoch)
                    plot_predictions(xx, yy, Z, x_train=x_train, x_test=x_test,
                                     y_train=y_train, y_test=y_test,
                                     title=title, plot_name=title)

    def calc_accuracy(self, test_dataloader):
        with torch.no_grad():
            n_correct = 0
            n_samples = 0

            for features, labels in test_dataloader:
                y_predicted = self.model(features)
                # max returns (value ,index)
                _, predicted = torch.max(y_predicted.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

            acc = 100.0 * n_correct / n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

    def predict(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)
        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)

                _, predicted = torch.max(output_batch.data, 1)
                logging.debug(predicted)

                all_outputs = torch.cat((all_outputs, predicted), 0)

        return all_outputs

    def predict_proba(self, test_dataloader):
        all_outputs = torch.tensor([], dtype=torch.long)

        self.model.eval()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(test_dataloader):
                output_batch = self.model(x_batch)
                all_outputs = torch.cat((all_outputs, output_batch), 0)

        return all_outputs

    def predict_proba_tensor(self, test_dataloader):
        self.model.eval()

        with torch.no_grad():
            output = self.model(test_dataloader)

        return output


def get_data_from_datasets(train_dataset, test_dataset):
    x_train = train_dataset.x_train.astype(np.float32)
    x_test = test_dataset.x_train.astype(np.float32)

    y_train = train_dataset.y_train.astype(int)
    y_test = test_dataset.y_train.astype(int)

    return x_train, x_test, y_train, y_test


def predict_proba_on_mesh_tensor(clf, xx, yy):
    q = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.predict_proba_tensor(q)[:, 1]
    Z = Z.reshape(xx.shape)
    return Z


def train(config: str, dataset: str):
    with open(config, 'r') as f:
        params = yaml.load(f)
        
    dataset = pd.read_csv(dataset)
    x_train, x_test, y_train, y_test = train_test_split(dataset.x, dataset.y, test_size=0.10, random_state=42)
    train_dataset = NumpyDataset(x_train, y_train)
    test_dataset = NumpyDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    learnig_rate = params['learning_rate']
    num_epochs = params['epochs']
    
    model = Network(2, params['network'], 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learnig_rate)
    
    trainer = Trainer(num_epochs, learnig_rate, model, criterion, optimizer)
    trainer.train(dataset, train_dataloader)
    trainer.calc_accuracy(test_dataloader)
