import logging
from typing import Literal
from uuid import uuid4

import click

from src.train import train as train_f
from src.datagen import datagen as datagen_f


def create_context():
    return {"run_id": str(uuid4())}


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


@cli.command()
@click.option(
    "--config", help="Configuration file", required=True,
)
@click.option(
    "--data", help="CSV data file", required=True,
)
def train(config, data):
    train_f(config, data)


@cli.command()
@click.option(
    "--name", help="Name of resulting file", required=True,
)
@click.option(
    "--data_type", help="Type of data to generate: moons/circles/blobs", required=True,
)
@click.option(
    "--size", help="Number of samples", required=True, type=int,
)
@click.option(
    "--random_seed", help="Random seed", required=False, type=int,
)
def datagen(name: str, data_type: Literal["moons", "circles", "blobs"], size: int, random_seed: int = 42):
    datagen_f(name=name, data_type=data_type, n_samples=size, random_seed=random_seed)


if __name__ == "__main__":
    cli()
