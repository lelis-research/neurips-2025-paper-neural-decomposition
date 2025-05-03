from config import arguments
from Scripts.train_agent import train_parallel_seeds
from Scripts.plot import load_results, plot_results, generate_video
from Scripts.test_agent import test_agent
from Scripts.tune_params import tune_ppo
from Scripts.learn_options import train_options, test_options
import os

if __name__ == "__main__":
    args = arguments()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    
    if "tune" in args.mode:
        tune_ppo(args)

    if "train" in args.mode:
        train_parallel_seeds(args.seeds, args)
    
    if "test" in args.mode:
        test_agent(args.test_seed, args)
            

    if "plot" in args.mode:
        results, folders = load_results(args)
        plot_results(results, 
                        window_size=args.smoothing_window_size, 
                        interpolation_resolution=args.interpolation_resolution,
                        nametag=f"{args.pattern}")
    
    if "train_option" in args.mode:
        train_options(args)

    if "test_option" in args.mode:
        test_options(args)

        
        
