import argparse

from jepa_fraudinha.data import prepare_data
from jepa_fraudinha.cli.types import AvailabeCommands


def build_prepare_data_cli(subparser: argparse._SubParsersAction) -> None:
    prepare_data_parser = subparser.add_parser(
        AvailabeCommands.PREPARE_DATA.value,
        help="Prepare datasets"
    )

    prepare_data_parser.add_argument(
        "--input",
        default="data/references.json",
        help="Path to references.json",
    )
    prepare_data_parser.add_argument(
        "--output",
        default="artifacts/data",
        help="Directory for prepared .npz files",
    )
    prepare_data_parser.add_argument("--seed", type=int, default=42)
    prepare_data_parser.add_argument("--train-size", type=float, default=0.70)
    prepare_data_parser.add_argument("--val-size", type=float, default=0.15)
    prepare_data_parser.add_argument("--test-size", type=float, default=0.15)

    prepare_data_parser.set_defaults(func=prepare_data)
