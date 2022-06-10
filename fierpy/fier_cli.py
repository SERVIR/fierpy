import argparse
import toml
from .fier_interface import build_regressions, run_synthesize

def main():

    parser = argparse.ArgumentParser(
        description="Forecasting of Inundation Extents using REOF (FIER) Command Line Application."
    )

    parser.add_argument( "-c", "--config", type=str, help="toml configuration file specifying the inputs for command", required=True)

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="which operation to run, options are 'buildregressions', 'synthesis",
        dest = "cmd",
        required = True
    )

    subparsers.add_parser('buildregressions')
    subparsers.add_parser('synthesis')

    args = parser.parse_args()

    config_dict = toml.load(args.config)

    if args.cmd == "buildregressions":
        build_regressions(config_dict)

    elif args.cmd == "synthesis":
        run_synthesize(config_dict)

    else:
        raise RuntimeError("wrong input")

    return 

if __name__ == "__main__":
    main()