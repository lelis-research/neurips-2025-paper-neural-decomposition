#!/usr/bin/env python
import os
import sys
import argparse
import importlib.util

from Scripts.train_agent import train_parallel_seeds
from Scripts.plot import load_results, plot_results, generate_video, plot_comparison
from Scripts.test_agent import test_agent
from Scripts.tune_params import tune_ppo
from Scripts.learn_options import train_options, test_options

def load_config_module(path: str):
    """Dynamically load a Python module from the given file path."""
    spec = importlib.util.spec_from_file_location("user_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiments based on a user-specified config.py"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to your config.py file",
    )
    args_cli = parser.parse_args()

    # load the config.py from the given path
    config_mod = load_config_module(args_cli.config_path)

    # instantiate the arguments dataclass
    args = config_mod.arguments()

    # make sure result directory exists
    os.makedirs(args.res_dir, exist_ok=True)

    # dispatch modes
    if "tune" in args.mode:
        tune_ppo(args)

    if "train" in args.mode:
        train_parallel_seeds(args.seeds, args)

    if "test" in args.mode:
        test_agent(args.test_seed, args)

    if "plot" in args.mode:
        if isinstance(args.pattern, str):
            results, folders = load_results(args)
            plot_results(
                results,
                window_size=args.smoothing_window_size,
                interpolation_resolution=args.interpolation_resolution,
                nametag=f"{args.pattern}",
            )
        elif isinstance(args.pattern, dict):
            plot_comparison(
                method_patterns=args.pattern,
                res_dir=args.res_dir,
                window_size=args.smoothing_window_size,
                interpolation_resolution=args.interpolation_resolution,
                out_fname=f"{args.plot_name}.png",
            )

    if "train_option" in args.mode:
        train_options(args)

    if "test_option" in args.mode:
        test_options(args)