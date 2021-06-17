# Pytorch Playground

Not recommended

### How to install

    pip install -r requirements.txt

### Generate data

    python cli.py datagen --name data.csv --data_type blobs --size 1000

### Run predictor

    python cli.py train --config config.yaml --data data.csv

#### Config

```YAML
epochs: 2
learning_rate: 0.1
network:
  - size: 5
    activation: relu
  - size: 5
    activation: tanh
```