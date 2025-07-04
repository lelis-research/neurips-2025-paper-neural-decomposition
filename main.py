#!/usr/bin/env python
import os
import sys
import argparse
import importlib.util

from Scripts.analyze_options import analyze_options
from Scripts.train_agent import train_parallel_seeds
from Scripts.plot import load_results, plot_results, generate_video, plot_comparison
from Scripts.test_agent import test_agent
from Scripts.tune_params import tune_agent
from Scripts.learn_options import train_options, test_options
from Scripts.search_options import search_options

def load_config_module(path: str):
    """
    Add the config's directory to sys.path and import the module by its filename.
    """
    path = os.path.abspath(path)
    config_dir, fname = os.path.split(path)
    module_name, ext = os.path.splitext(fname)
    if module_name == "" or ext.lower() != ".py":
        raise ValueError(f"{path} is not a Python file")
    # make it importable
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    return importlib.import_module(module_name)

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
    print(args)

    # make sure result directory exists
    os.makedirs(args.res_dir, exist_ok=True)

    # dispatch modes
    if "tune" in args.mode:
        tune_agent(args)

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
                out_fname=f"{args.image_base_dir}/{args.plot_name}.svg",
            )

    if "search_option" in args.mode:
        search_options(args)

    if "train_option" in args.mode:
        train_options(args)

    if "analyze_option" in args.mode:
        analyze_options(args)
        
    if "test_option" in args.mode:
        test_options(args)