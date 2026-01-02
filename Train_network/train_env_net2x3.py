#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
from config import get_config
from runners.separated.runner import CRunner as Runner
import copy # [ADD] Thêm thư viện copy để xử lý config

# Import module wrapper để thực hiện patch
import envs.env_wrappers as env_wrappers_module
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv

# ==============================================================================
# 1. DYNAMIC IMPORT & PATCHING
# ==============================================================================
parser = get_config()
all_args, unknown = parser.parse_known_args(sys.argv[1:])

if all_args.scenario_name == 'Inventory_Management_Network':
    from envs.net_2x3 import Env
    print(f"--> [SYSTEM] Đang sử dụng môi trường NETWORK (2-2-2) từ envs/net_2x3.py")
else:
    from envs.serial import Env
    print(f"--> [SYSTEM] Đang sử dụng môi trường SERIAL (1-1-1) từ envs/serial.py")

# [MAGIC FIX] Gán đè class Env vào module wrapper
env_wrappers_module.Env = Env

# ==============================================================================
# 2. HÀM TẠO MÔI TRƯỜNG (FIX TRIỆT ĐỂ)
# ==============================================================================
def make_train_env(all_args):
    # Training dùng đúng n_rollout_threads (ví dụ: 5)
    print(f"--> [SYSTEM] Init {all_args.n_rollout_threads} Train Envs (DummyVecEnv Mode)...")
    return DummyVecEnv(all_args)

def make_eval_env(all_args):
    # [FIX QUAN TRỌNG] 
    # DummyVecEnv mặc định đọc 'n_rollout_threads' để tạo env.
    # Khi Eval, ta cần nó đọc 'n_eval_rollout_threads' (thường là 1).
    # Ta phải tạo một bản copy của config và tráo đổi giá trị này.
    eval_args = copy.copy(all_args)
    eval_args.n_rollout_threads = all_args.n_eval_rollout_threads
    
    print(f"--> [SYSTEM] Init {eval_args.n_rollout_threads} Eval Envs (DummyVecEnv Mode)...")
    return DummyVecEnv(eval_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == "__main__":
    all_args = parse_args(sys.argv[1:], parser)
    
    seeds = [all_args.seed]
    if isinstance(seeds, int):
        seeds = [seeds]
    elif isinstance(seeds, (list, tuple)) and len(seeds) == 1 and isinstance(seeds[0], (list, tuple)):
        seeds = list(seeds[0])
        
    print("all config: ", all_args)

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

        curr_run = 'run_seed_%i' % (seed)

        seed_res_record_file = run_dir / "seed_results.txt"
        
        curr_run_dir = run_dir / curr_run
        curr_run_dir.mkdir(parents=True, exist_ok=True)
        # (curr_run_dir / "models").mkdir(parents=True, exist_ok=True) # Để runner tự lo việc tạo folder này

        if not seed_res_record_file.exists():
            seed_res_record_file.touch()

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

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

        runner = Runner(config)
        
        reward, bw, cost, service = runner.run()

        with open(seed_res_record_file, 'a+') as f:
            f.write(str(seed) + ' ' + str(reward) + ' ')
            for fluc in bw:
                f.write(str(fluc) + ' ')
            for c in cost:
                f.write(str(c) + ' ')
            for s in service:
                f.write(str(s) + ' ')
            f.write('\n')

        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()