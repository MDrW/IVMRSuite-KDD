import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import random
from ecs_metaheuristic.aco.ecs_aco_env import ECSACOEnvironment
from multiprocessing import Process, Queue, Pool
from argparse import ArgumentParser
import time
from copy import deepcopy as dcp
from scipy import sparse
import json
import pdb


def single_ant_explore(args, pheromone_matrix, pheromone_matrix_numa, seed, is_eval=False):
    env = ECSACOEnvironment(data_path=args.data_path, seed=seed, is_process_numa=True)
    np.random.seed(seed)
    random.seed(seed)
    if np.random.uniform(0, 1) < 0.5: 
        item_idxes = np.arange(env.item_count)
        np.random.shuffle(item_idxes)
    else:
        item_idxes = env.item_sorted_idxes
    exp_i = 0
    while (len(item_idxes) > 0) and (exp_i < 100):
        for i, item_i in enumerate(item_idxes):
            # print(i, item_i)
            available_boxes = env.get_item_available_actions(item_i)
            # pdb.set_trace()
            if not available_boxes.any():
                available_boxes = (env.init_actions_available[item_i] == 1)
            heuristic_eta = env.get_item_rewards(item_i)[available_boxes]
            # heuristic_eta = np.power((heuristic_eta - heuristic_eta.min() + 1) / (heuristic_eta.max() - heuristic_eta.min() + 1), args.beta)
            heuristic_eta = np.power((heuristic_eta - heuristic_eta.min() + 1), args.beta)
            pheromone_tao = np.power((pheromone_matrix[item_i] - pheromone_matrix[item_i].min() + 1)[available_boxes], args.alpha)
            # pdb.set_trace()
            # pheromone_tao = np.power((pheromone_tao - pheromone_tao.min() + 1) / (pheromone_tao.max() - pheromone_tao.min() + 1), args.alpha)
            # print(heuristic_eta)
            # print(pheromone_tao)
            prob_val = heuristic_eta * pheromone_tao
            prob = np.zeros(len(available_boxes), dtype=float)
            prob[available_boxes] = prob_val / prob_val.sum()
            # print(prob_val / prob_val.sum())
            if is_eval:
                box_k = np.argmax(prob)
            else:
                box_k = np.random.choice(np.arange(env.box_count), replace=False, p=prob)
            # box_k = np.where(env.init_placement[item_i] == 1)[0][0]
            # numa_action = (env.init_items_numa[env.item_init_movable][item_i].reshape(env.max_box_numa, -1) > 0).any(axis=-1).astype(int)
            # numa_action = stored_numa_actions.get((item_i, box_k))
            # numa_action = None
            numa_available = env.get_feasible_numa_actions(item_i, box_k)
            if not numa_available.all():
                numa_available = np.ones(env.max_box_numa, dtype=bool)
            heuristic_numa_eta = np.power(env.get_numa_actions_heuristic(item_i, box_k)[numa_available], args.beta)
            pheromone_numa_tao = np.power((pheromone_matrix_numa[item_i, box_k] - pheromone_matrix_numa[item_i, box_k].min() + 1)[numa_available], args.alpha)
            prob_numa_val = heuristic_numa_eta * pheromone_numa_tao
            prob_numa = np.zeros(len(numa_available), dtype=float)
            prob_numa[numa_available] = prob_numa_val / prob_numa_val.sum()
            # print(heuristic_numa_eta, pheromone_numa_tao, prob_numa)
            if is_eval:
                numa_action_idx = np.argsort(prob_numa)[::-1][:env.item_numa_num[item_i]]
                
            else:
                numa_action_idx = np.random.choice(np.arange(len(prob_numa)), replace=False, p=prob_numa, size=env.item_numa_num[item_i])
            numa_action = np.zeros(len(prob_numa), dtype=int)
            numa_action[numa_action_idx] = 1
            env.step([item_i, box_k], numa_action=numa_action)
            # env.recovery_to_feasible_placement_by_boxes()
        # pdb.set_trace()    
        recovery_iter, recovery_flag = 0, False
        simu_env = dcp(env)
        while (not recovery_flag) and (recovery_iter <= 3):
            recovery_flag = simu_env.recovery_to_feasible_placement_by_boxes()
            recovery_iter += 1
        env = simu_env
        if recovery_flag:
            # env = simu_env
            break
        item_idxes = env.recovery_to_feasible_placement_by_items()
        exp_i += 1
        # pdb.set_trace()
    # print(exp_i)
    # pdb.set_trace()
    if (not recovery_flag) and (np.random.uniform(0, 1) < 0.2):
        env.recovery_to_init_placement()
    total_reward = env.get_final_reward()
    final_cost = env.get_final_real_cost()
    final_place = env.get_final_placement()
    final_numa_actions = env.get_stored_numa_actions()
    return total_reward, final_cost, final_place, final_numa_actions

def _write_info(info, logger=None):
    if logger is None:
        print(info)
    else:
        logger.write((info+'\n').encode())
        logger.flush()


def run_ant_colony_optimization(args):
    if os.path.exists(args.log_save_path):
        logger = open(os.path.join(args.log_save_path, args.data_name+'.log'), 'wb')
    else:
        logger = None
    _write_info(f"Start Running Ant Colony Optimization on {args.data_path}...", logger)
    start = time.time()
    env = ECSACOEnvironment(data_path=args.data_path, is_process_numa=True)
    pheromone_matrix = np.ones((env.item_count, env.box_count), dtype=float) * 10
    pheromone_matrix_numa = np.ones((env.item_count, env.box_count, env.max_box_numa), dtype=float)
    stored_numa_actions = {}
    blocked_count = 0
    global_best_cost, global_best_rew, global_best_placement, global_best_rew_placement = np.inf, -np.inf, None, None
    global_best_numa_actions = {}
    for ei in range(args.epoches):
        st = time.time()
        # print(f"Running Epoch {ei}...")
        pool = Pool(processes=args.num_process)
        results = []
        for ai in range(args.n_ants):
            # single_ant_explore(args, pheromone_matrix, ai+(ei+1)*10+args.seed, False,)
            results.append(pool.apply_async(single_ant_explore, args=(args, pheromone_matrix, pheromone_matrix_numa, ai*(ei+1)+args.seed, (ei%2)==0, )))
        pool.close()
        pool.join()
        # pheromone_matrix *= (1 - args.rho)
        if ei % args.lr_interval == 0:
            args.alpha = np.clip(args.alpha * (1 + args.lr_update), 1.0, 5.0)
            args.beta = np.clip(args.beta * (1 - args.lr_update), 0.2, 1.0)
        
        best_rew, best_cost, tao_update, this_placement, numa_actions_update = -np.inf, np.inf, None, None, {}
        for ai in range(args.n_ants):
            res = results[ai].get()
            if res[0] > best_rew:
                best_rew = res[0]
                best_cost = res[1]
                tao_update = res[2].copy().astype(float)
                numa_actions_update = res[3]
                numa_actions_update[numa_actions_update > 0] = res[0]
                tao_update[tao_update > 0] = res[0]
                this_placement = res[2]
        # pdb.set_trace()
        update_ratio = args.update_ratio
        pheromone_matrix = (1 - args.rho) * pheromone_matrix + update_ratio * tao_update
        # stored_numa_actions.update(numa_actions_update)
        pheromone_matrix_numa = (1 - args.rho) * pheromone_matrix_numa + update_ratio * numa_actions_update * 0.01
        # print(pheromone_matrix_numa.max(), pheromone_matrix_numa.min())

        _write_info(f"@Epoch{ei}, Best Reward: {best_rew}, Best Cost: {best_cost}, Spend Time: {time.time()-st}s!", logger)
        
        relative_cost_gap = (global_best_cost - best_cost) / (abs(best_cost) + 1e-5) if not np.isinf(best_cost) else np.inf
        relative_gap = (best_rew - global_best_rew) / (abs(best_rew) + 1e-5)
        if best_cost < global_best_cost:
            global_best_placement, global_best_cost = this_placement, best_cost
            # global_best_numa_actions = dcp(numa_actions_update)
            _write_info(f"* Find Better Placement, Current Best Cost: {best_cost}, Relative Gap {relative_cost_gap}!", logger)
        if best_rew > global_best_rew:
            if best_rew > 0:
                global_best_rew_placement, global_best_rew = this_placement, best_rew
                global_best_numa_actions = dcp(numa_actions_update).reshape(len(numa_actions_update), -1)
            else:
                global_best_rew_placement, global_best_rew = env.init_placement, 0.
                global_best_numa_actions = dcp(env.get_stored_numa_actions()).reshape(len(numa_actions_update), -1)
            sparse.save_npz(os.path.join(args.save_path, args.data_name+'_aco_placement.npz'), sparse.csr_matrix(global_best_rew_placement))
            sparse.save_npz(os.path.join(args.save_path, args.data_name+'_aco_numa.npz'), sparse.csr_matrix(global_best_numa_actions))
            # store numa_actions
            _write_info(f"* Find Better Placement, Current Best Reward: {best_rew}, Relative Gap {relative_gap}!", logger)
        if (relative_cost_gap < args.rel_gap) or (np.isinf(relative_cost_gap) and (not np.isinf(global_best_cost))):
            blocked_count += 1
            if blocked_count >= args.blocked_limit:
                _write_info(f"! Relative Gap {relative_gap} is small, Stop Iteration!", logger)
                break
        else:
            blocked_count = 0
        
        if (time.time() - start) > args.time_limit:
            _write_info(f"! Time Limit {args.time_limit}s Exceed!", logger)
            break
    # sparse.save_npz(os.path.join(args.save_path, args.data_name+'_aco_placement.npz'), sparse.csr_matrix(global_best_rew_placement))
    _write_info(f'Finish Solved {args.data_path} by Ant Colony Optimization! Best Cost: {global_best_cost}, Best Reward: {global_best_rew}, Total Time: {time.time()-start}s!', logger)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root_path', type=str, default='../../data/ecs_data')
    parser.add_argument('--data_name', type=str, default='0')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr_interval', type=int, default=5)
    parser.add_argument('--lr_update', type=float, default=0.2)
    
    parser.add_argument('--rho', type=float, default=0.001)
    parser.add_argument('--update_ratio', type=float, default=0.01)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('--num_process', type=int, default=64)
    parser.add_argument('--n_ants', type=int, default=64)
    parser.add_argument('--seed', type=int, default=19753)
    parser.add_argument('--blocked_limit', type=int, default=10)
    parser.add_argument("--rel_gap", type=float, default=1e-3)
    parser.add_argument('--time_limit', type=int, default=7200)
    
    parser.add_argument("--save_path", type=str, default='../../results/ecs_metaheuristic_aco/placement')
    parser.add_argument("--log_save_path", type=str, default='../../results/ecs_metaheuristic_aco/logs')
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_save_path, exist_ok=True)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    with open('../../data/ecs_data_classification.json', 'r') as f:
        data_classified = json.load(f)
        data_list = data_classified.values()
    
    for datas in data_list:
        for data in datas:
            if not os.path.exists(os.path.join(args.save_path, f"{data}_aco_placement.npz")):
                args.data_name = data
                args.data_path = os.path.join(args.data_root_path, args.data_name)
                run_ant_colony_optimization(args=args)
