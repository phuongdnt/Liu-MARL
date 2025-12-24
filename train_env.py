#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runners.separated.runner import CRunner as Runner

def make_train_env(all_args):
    return SubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DummyVecEnv(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    seeds = [all_args.seed]
    if isinstance(seeds, int):
        seeds = [seeds]
    elif isinstance(seeds, (list, tuple)) and len(seeds) == 1 and isinstance(seeds[0], (list, tuple)):
        seeds = list(seeds[0])
        
    print("all config: ", all_args)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    for seed in seeds:
        print("-------------------------------------------------Training starts for seed: " + str(seed)+ "---------------------------------------------------")

        project_dir = Path(__file__).resolve().parent
        results_dir = project_dir.parent / "results"
        run_dir = results_dir / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)

        curr_run = 'run_seed_%i' % (seed + 1)

        seed_res_record_file = run_dir / "seed_results.txt"
        
        curr_run_dir = run_dir / curr_run
        curr_run_dir.mkdir(parents=True, exist_ok=True)
        (curr_run_dir / "models").mkdir(parents=True, exist_ok=True)  

        if not seed_res_record_file.exists():
            seed_res_record_file.touch()

        # seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # env
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        num_agents = all_args.num_agents

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": curr_run_dir
        }

        # run experiments
        runner = Runner(config)
        reward, bw = runner.run()

        with open(seed_res_record_file, 'a+') as f:
            f.write(str(seed) + ' ' + str(reward) + ' ')
            for fluc in bw:
                f.write(str(fluc) + ' ')
            f.write('\n')

        # post process
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()

    
