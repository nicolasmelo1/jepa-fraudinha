from jepa_fraudinha.cli import build_cli


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
