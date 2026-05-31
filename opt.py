import argparse


def build_argument_parser():
    cli_parser = argparse.ArgumentParser(
        description="Configuration settings for graph clustering training"
    )

    training_options = {
        "--gnnlayers_h": {
            "type": int,
            "default": 3,
            "help": "Number of high-frequency GNN layers"
        },
        "--gnnlayers_l": {
            "type": int,
            "default": 3,
            "help": "Number of low-frequency GNN layers"
        },
        "--epochs": {
            "type": int,
            "default": 400,
            "help": "Total number of training epochs"
        },
        "--dims": {
            "type": int,
            "nargs": "+",
            "default": [1000],
            "help": "Hidden layer dimensions"
        },
        "--lr": {
            "type": float,
            "default": 1e-3,
            "help": "Learning rate"
        },
        "--sigma": {
            "type": float,
            "default": 0.01,
            "help": "Gaussian noise standard deviation"
        },
        "--dataset": {
            "type": str,
            "default": "citeseer",
            "help": "Dataset name"
        },
        "--cluster_num": {
            "type": int,
            "default": 7,
            "help": "Number of clusters"
        },
        "--device": {
            "type": str,
            "default": "cuda:0",
            "help": "Computation device"
        },
        "--gama": {
            "type": float,
            "default": 0.5,
            "help": "Loss balancing coefficient"
        },
        "--sigma_X": {
            "type": float,
            "default": 0.2,
            "help": "Feature-level Gaussian noise standard deviation"
        }
    }

    for argument_name, argument_config in training_options.items():
        cli_parser.add_argument(argument_name, **argument_config)

    return cli_parser


args = build_argument_parser().parse_args()