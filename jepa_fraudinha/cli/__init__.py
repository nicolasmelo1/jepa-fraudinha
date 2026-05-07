import argparse

from jepa_fraudinha.cli.prepare_data import build_prepare_data_cli
from jepa_fraudinha.cli.train import build_train_cli


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            description="Train and run a JEPA-like model, used for learning "
                        "purposes"
    )

    subparsers = parser.add_subparsers(
            dest="command",
            required=True
    )
    build_prepare_data_cli(subparsers)
    build_train_cli(subparsers)

    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
