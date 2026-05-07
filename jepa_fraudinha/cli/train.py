import argparse

from jepa_fraudinha.cli.types import AvailabeCommands, Models
from jepa_fraudinha.training.baseline.knn import train_knn
from jepa_fraudinha.training.baseline.logistic import train_logistic
from jepa_fraudinha.training.baseline.mlp import train_mlp


def build_train_logistic_cli(subparser: argparse._SubParsersAction) -> None:
    train_logistic_parser = subparser.add_parser(
        Models.LOGISTIC.value,
        help="Train Logistic model"
    )

    train_logistic_parser.add_argument(
        "--data-folder",
        default="artifacts/data",
        help="Path to train.npz",
    )
    train_logistic_parser.add_argument(
        "--metrics-folder",
        default="artifacts/metrics",
        help="Directory for metrics JSON files",
    )
    train_logistic_parser.set_defaults(func=train_logistic)


def build_train_mlp_cli(subparser: argparse._SubParsersAction) -> None:
    train_mlp_parser = subparser.add_parser(
        Models.MLP.value,
        help="Train MLP baseline",
    )

    train_mlp_parser.add_argument(
        "--data-folder",
        default="artifacts/data",
        help="Directory containing train.npz, val.npz, and test.npz",
    )

    train_mlp_parser.add_argument(
        "--metrics-folder",
        default="artifacts/metrics",
        help="Directory for metrics JSON files",
    )
    train_mlp_parser.add_argument("--hidden-dim", type=int, default=64)
    train_mlp_parser.add_argument("--epochs", type=int, default=5)
    train_mlp_parser.add_argument("--batch-size", type=int, default=8192)
    train_mlp_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_mlp_parser.add_argument("--seed", type=int, default=42)

    train_mlp_parser.set_defaults(func=train_mlp)


def build_train_knn_cli(subparser: argparse._SubParsersAction) -> None:
    train_knn_parser = subparser.add_parser(
        Models.KNN.value,
        help="Train KNN baseline",
    )

    train_knn_parser.add_argument(
        "--data-folder",
        default="artifacts/data",
        help="Directory containing train.npz, val.npz, and test.npz",
    )
    train_knn_parser.add_argument(
        "--metrics-folder",
        default="artifacts/metrics",
        help="Directory for metrics JSON files",
    )
    train_knn_parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[5, 11, 31],
        help="K values to evaluate",
    )
    train_knn_parser.add_argument(
        "--train-sample",
        type=int,
        default=100_000,
        help="Number of train rows to use as KNN references",
    )
    train_knn_parser.add_argument(
        "--eval-sample",
        type=int,
        default=50_000,
        help="Number of rows to evaluate per val/test split",
    )
    train_knn_parser.add_argument(
        "--metric",
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Nearest-neighbor distance metric",
    )
    train_knn_parser.add_argument(
        "--batch-size",
        type=int,
        default=1_000,
        help="Rows to score per nearest-neighbor query batch",
    )
    train_knn_parser.add_argument("--seed", type=int, default=42)

    train_knn_parser.set_defaults(func=train_knn)


def build_train_cli(subparser: argparse._SubParsersAction) -> None:
    train_parser = subparser.add_parser(
        AvailabeCommands.TRAIN.value,
        help="Train models"
    )
    train_subparser = train_parser.add_subparsers(
        dest="model",
        required=True
    )

    build_train_logistic_cli(train_subparser)
    build_train_mlp_cli(train_subparser)
    build_train_knn_cli(train_subparser)
