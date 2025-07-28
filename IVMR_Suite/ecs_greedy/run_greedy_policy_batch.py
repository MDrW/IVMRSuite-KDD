import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import pandas as pd
from ecs_env.ecs_base_env import ECSEnvironment
from multiprocessing import Pool
from ecs_greedy.greedy_policy import greedy_policy
from ecs_greedy.greedy_log_parse import parse_logger


def run_greedy(data_path, logger_path, seed, max_bad_action_count):
    logger = open(logger_path, 'wb')
    logger.write(f"Start Running {data_path}...\n".encode())
    logger.flush()
    env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    init_score = env.get_cur_state_score()
    
    logger.write(f"Initial State Score: {init_score}\n".encode())
    logger.flush()
    start = time.time()
    action_list = greedy_policy(ecs_env=env, logger=logger, seed=seed, max_bad_action_count=max_bad_action_count)
    end = time.time()
    
    eval_reward, cost_infos, num_inv = env.eval_move_sequence(action_list)
    des_score = env.get_cur_state_score()
    logger.write(f"Eval Reward: {eval_reward}\n".encode())
    mig_cost, eval_score = cost_infos['migration_cost'], des_score+cost_infos['migration_cost']-init_score
    logger.write(f"Migration Cost: {mig_cost}, Finish Score: {des_score}, Total Eval Score: {eval_score}\n".encode())
    logger.write(f"Elapsed Time: {end-start}s.\n".encode())
    
    logger.write(f"End Running {data_path}...\n".encode())
    logger.flush()


if __name__ == '__main__':
    data_type = 'ecs_data'
    num_threads = 32
    seed = 15937
    max_bad_action_count = 0
    root_path = f'../data/{data_type}'
    logger_path = f'../results/ecs_greedy/logs/greedy_{data_type}_{max_bad_action_count}_{seed}'
    result_save_path = f'../results/ecs_greedy/results/greedy_{data_type}_{max_bad_action_count}_{seed}.csv'
    
    os.makedirs(logger_path, exist_ok=True)
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    
    files = os.listdir(root_path)
    pool = Pool(processes=num_threads)
    for file in files:
        pool.apply_async(run_greedy, args=(os.path.join(root_path, file), os.path.join(logger_path, f"{file}.log"),
                                           seed, max_bad_action_count,))
    pool.close()
    pool.join()
    
    parse_logger(
        logger_root_path=logger_path,
        result_save_path=result_save_path,
    )
