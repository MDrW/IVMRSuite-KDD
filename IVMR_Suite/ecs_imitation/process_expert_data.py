import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy import sparse
import os
from ecs_env.ecs_env_binary_wrapper import ECSRawDenseEnvironment
import pickle as pkl
from multiprocessing import Pool
import pdb
import time
from copy import deepcopy as dcp


def parse_logger(logger_path):
    with open(logger_path, 'r') as f:
        lines = f.readlines()
    
    move_seq, numa_seq = [], []
    for line in lines:
        if (line.startswith('Step') or line.startswith('Sorted Step') or line.startswith('[Break Cycle') or line.startswith('[Undo ') or line.startswith('[Resort and Reindex] Step')) and ('can move from' in line):
            if line.startswith('Sorted Step 1:') or line.startswith('[Resort and Reindex] Step 1:'):
                move_seq, numa_seq = [], []
            # print(line, line.split('Numa [')[1].split(']!')[0].strip().split(','))
            item = int(line.split('Item ')[1].split(' can move')[0].strip())
            if 'Numa' in line:
                box_j = int(line.split('from Box ')[1].split(' Numa')[0].strip())
                box_k = int(line.split('to Box ')[1].split(' Numa')[0].strip())
                numa = line.split('Numa [')[1].split(']!')[0].strip()
                if numa != '':
                    numa = [int(n) for n in numa.split(' ')]
                else:
                    numa = []
            else:
                box_j = int(line.split('from Box ')[1].split(' to Box')[0].strip())
                box_k = int(line.split('to Box ')[1].split(' ')[0].strip())
                numa = []
            rew = float(line.split(': ')[1].split(', ')[0].strip())
            if (line.startswith('[Undo ')):
                for j in range(len(move_seq)-1, -1, -1):
                    if (move_seq[j][0] == item) and (move_seq[j][1] == box_j) and (move_seq[j][2] == box_k):
                        move_seq = move_seq[:j] + move_seq[j+1:]
                        numa_seq = numa_seq[:j] + numa_seq[j+1:]
                        break
            else:
                move_seq.append([item, box_k, box_j, rew])
                numa_seq.append(numa)
    return move_seq, numa_seq

def get_expert_data(data_path, logger_path, save_path, is_dense_items, is_dense_boxes, is_state_merge_placement, is_filter_unmovable,
                    is_origin_filter_unmovable, is_process_numa, comp_reward=None, logger=None):
    _log(f"Start Parse Expert Data {logger_path}...", logger)
    # print(f"Start Parse Expert Data {logger_path}...")
    ecs_env = ECSRawDenseEnvironment(data_path=data_path, is_limited_count=False, is_filter_unmovable=is_filter_unmovable,
                                     is_dense_items=is_dense_items, is_dense_boxes=is_dense_boxes, is_state_merge_placement=is_state_merge_placement,
                                     is_process_numa=is_process_numa)
    # print("ECS Environment Init Done!")
    move_seq, numa_seq = parse_logger(logger_path=logger_path)
    # pdb.set_trace()
    rewards = np.array(move_seq)[:, -1].tolist()
    move_seq = np.array(move_seq)[:, :2].astype(int).tolist()
    # print("Parse Logger Done!")
    # pdb.set_trace()
    state = ecs_env.reset()
    results = []
    times, total_reward, total_real_reward = 0, 0, 0
    total_invalid_actions = 0
    # pre_actions = []
    _log(f"Dense Environment: Items from {ecs_env.item_count} to {len(ecs_env.dense_items_map)}, Boxes from {ecs_env.box_count} to {len(ecs_env.dense_boxes_map)}!", logger)
    
    while len(move_seq) > 0:
        _log(f"Start Iter {times}...", logger)
        new_move_seq, new_rewards = [], []
        for mi, action in enumerate(move_seq):
            s1 = time.time()
            if not is_origin_filter_unmovable:
                action[0] = ecs_env.ecs_env.init_idxes_map[action[0]]
            
            if np.asarray(state[-2].todense())[ecs_env.dense_items_idxes[action[0]], ecs_env.dense_boxes_idxes[action[1]]] <= 0:
                new_move_seq.append(action)
                new_rewards.append(rewards[mi])
                _log(f"Iter {times}: Action {list(action)} Invalid!", logger)
                continue
            # next_state, reward, done, info = ecs_env.step(list(action))
            # pact = [ecs_env.dense_items_idxes[action[0]], ecs_env.dense_boxes_idxes[ecs_env.ecs_env.get_item_cur_box[action[0]]], ecs_env.dense_boxes_idxes[action[1]]]
            # mix_cost, vio_cost = ecs_env._get_cost_matrix()
            if len(numa_seq[mi]) > 0:
                numa_action = np.zeros(ecs_env.ecs_env.max_box_numa, dtype=int)
                numa_action[numa_seq[mi]] = 1
            else:
                numa_action = None
            next_state, reward, done, info = ecs_env.step([ecs_env.dense_items_idxes[action[0]], ecs_env.dense_boxes_idxes[action[1]]], is_act_dense=True, numa_action=numa_action)
            _log(f"Step {info['move_count']}: reward {reward}, real reward {info['real_reward']}, done {done}, action {action}, dense action {[ecs_env.dense_items_idxes[action[0]], ecs_env.dense_boxes_idxes[action[1]]]}, {info['action_info']}, {time.time()-s1}s.", logger)
            # _log(f"Step {info['move_count']}: reward {reward}, real reward {info['real_reward']}, done {done}, {info['action_info']}, {time.time()-s1}s.", logger)
            _log(f"SVio {info['move_count']}: {info['vio_info']}", logger)
            total_reward += reward
            total_real_reward += info['real_reward']
            
            results.append([*dcp(state[:3]), [ecs_env.dense_items_idxes[action[0]], ecs_env.dense_boxes_idxes[action[1]]], reward, done, *dcp(state[-3:])])
            # pre_actions.append(pact)
            state = next_state

        times += 1
        move_seq = new_move_seq
        rewards = new_rewards
        if times <= 1:
            total_invalid_actions = len(move_seq)
        
        _log(f"Finish Iter {times}, Undo Action Num {len(new_move_seq)}, Undo Reward {np.sum(rewards)}!", logger)
        if (np.sum(rewards) <= 0) or (times > total_invalid_actions):
            break
    
    _log(f"Total Reward: {total_reward}, Total Real Reward: {total_real_reward}, Num Move: {info['move_count']}, Final Score: {ecs_env.get_cur_state_score()}!", logger)
    
    if (comp_reward is not None) and (np.abs(total_real_reward - comp_reward) > 1):
        _log(f"{logger_path} Comp Reward {comp_reward} is not equal to Real Reward {total_real_reward}! Exit!", logger)
        # return False
    # pdb.set_trace()
    # results = [results, *state[3:]]
    os.makedirs(save_path, exist_ok=True)
    for i, r in enumerate(results):
        with open(os.path.join(save_path, f'{i}.pkl'), 'wb') as f:
            pkl.dump(r, f)
    
    with open(os.path.join(save_path, 'edge_feats.pkl'), 'wb') as f:
        res = state[3:5]
        pkl.dump(res, f)
    _log(f"Finish Parse Expert Data {logger_path}...", logger)
    _log("-------------------------------------------\n", logger)
    return True


def compare_and_choose_result(comp_result_path_list, rename_cols_list, comp_cols, logger=None):
    comp_result = None
    for ri, result_path in enumerate(comp_result_path_list):
        cols = rename_cols_list[ri]
        result = pd.read_csv(result_path, sep=',', header=0)
        result['name'] = result['name'].astype(str).apply(lambda x: x.split('/')[-1])
        if cols is not None and type(cols) == dict:
            result = result.rename(columns=cols)[['name'] + list(cols.values())]
        if comp_result is not None:
            comp_result = pd.merge(comp_result, result, on=['name'], how='inner')
            change_flag = comp_result['use_reward'] <= comp_result[comp_cols[ri]]
            comp_result.loc[change_flag, 'use_idx'] = ri
            comp_result.loc[change_flag, 'use_reward'] = comp_result.loc[change_flag, comp_cols[ri]].values
        else:
            comp_result = result
            comp_result['use_idx'] = ri
            comp_result['use_reward'] = comp_result[comp_cols[ri]].values
        # pdb.set_trace()
    
    filter_flag = (comp_result['use_reward']) > 0
    stored_comp_result = comp_result[filter_flag]
    _log(f"There are {filter_flag.sum()} cases to process! {(~filter_flag).sum()} No-Move Cases have been drop!", logger)
    if (~filter_flag).sum() > 0:
        _log(f"Droped Cases: {comp_result.loc[~filter_flag, 'name'].apply(lambda x: x.split('/')[-1]).values.tolist()}!", logger)
    return stored_comp_result.reset_index()


def _log(info, logger=None):
    if logger is None:
        print(info)
    else:
        logger.write(info+'\n')
        logger.flush()


if __name__ == '__main__':
    is_regenerate = False
    is_dense_items, is_dense_boxes = False, False
    is_filter_unmovable, is_state_merge_placement, is_process_numa = True, True, True
    dense_name = 'expert_data_dense_items_boxes'
    if is_dense_items and (not is_dense_boxes):
        dense_name = 'expert_data_dense_items'
    elif (not is_dense_items) and (not is_dense_boxes):
        dense_name = 'expert_data'
    elif (not is_dense_items):
        raise NotImplementedError
    if is_state_merge_placement:
        dense_name += '_merge_placement'
    
    data_type = 'ecs_data'
    data_root_path = f'../data/{data_type}'
    root_path = '../results/ecs_imitation'
    os.makedirs(root_path, exist_ok=True)
    
    root_logger = open(f'{root_path}/get_{dense_name}_{data_type}.log', 'w')
    
    logger_root_path_list = [
        f'../results/ecs_opt/opt_logs/{data_type}_addmig1.0_move_seq',
        f'../results/ecs_greedy/logs/greedy_{data_type}_0_15937',
    ]
    comp_result_path_list = [
        f'../results/ecs_opt/results/opt_{data_type}.csv',
        f'../results/ecs_greedy/results/greedy_{data_type}_0_15937.csv',
    ]
    
    rename_cols_list = [
        {'sorted_total_reward': 'opt_reward', 'sorted_final_score': 'opt_final_cost', 'sorted_move_count': 'opt_move_count'},
        {'eval_reward': 'greedy_reward', 'final_cost': 'greedy_final_cost', 'action_count': 'greedy_move_count'},
    ]
    
    comp_cols = ['opt_reward', 'greedy_reward']
    comp_result = compare_and_choose_result(
        comp_result_path_list=comp_result_path_list,
        rename_cols_list=rename_cols_list,
        comp_cols=comp_cols,
        logger=root_logger,
    )
    
    save_root_path = f'{root_path}/{dense_name}/{data_type}'
    os.makedirs(save_root_path, exist_ok=True)
    logger_root_path = f"{root_path}/{dense_name}_logs/{data_type}"
    os.makedirs(logger_root_path, exist_ok=True)
    
    def _task(i):
        data_name = comp_result.loc[i, 'name'].split('/')[-1]
        use_idx = int(comp_result.loc[i, 'use_idx'])
        logger_path = f'{logger_root_path_list[use_idx]}/{data_name}.log'
        _log(f"Processing Task {i} {data_name}...", root_logger)
        if not os.path.exists(logger_path):
            _log(f"Task {i} logger path {logger_path} does not exist!", root_logger)
            return
        
        save_path = f'{save_root_path}/{data_name}'
        data_path = f'{data_root_path}/{data_name}'
        
        if os.path.exists(save_path) and (not is_regenerate):
            _log(f"Exist {data_name}! No ReGenerating for {data_name}! Exit Task {i}!", root_logger)
        else: 
            _log(f"Try Processing Task {i} {data_name}...", root_logger)
            logger = open(f'{logger_root_path}/{data_name}.log', 'w')
            
            comp_reward = float(comp_result.loc[i, 'use_reward'])
            _log(f"Use Policy {use_idx}: {logger_path}! Total Reward: {comp_reward}!", logger)
            is_origin_filter_unmovable = True
            is_success = get_expert_data(
                data_path=data_path,
                logger_path=logger_path,
                save_path=save_path,
                is_dense_items=is_dense_items,
                is_dense_boxes=is_dense_boxes,
                is_state_merge_placement=is_state_merge_placement,
                is_filter_unmovable=is_filter_unmovable,
                is_process_numa=is_process_numa,
                is_origin_filter_unmovable=is_origin_filter_unmovable,
                comp_reward=comp_reward,
                logger=logger,
            )
            _log(f"Finish Processed {data_name}, Successed Task {i}: {is_success}...", root_logger)
    
    # _task(183)
    print(len(comp_result))
    pool = Pool(processes=4)
    for i in range(len(comp_result)):
        pool.apply_async(_task, args=(i,))
    pool.close()
    pool.join()
