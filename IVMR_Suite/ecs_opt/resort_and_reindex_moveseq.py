import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy import sparse
import os
from ecs_env.ecs_base_env_binary import ECSEnvironment
from ecs_env.ecs_env_binary_wrapper import ECSRawDenseEnvironment
import pickle as pkl
from multiprocessing import Pool
import pdb
import time
import networkx as nx
from copy import deepcopy as dcp


def parse_logger(logger_path):
    with open(logger_path, 'r') as f:
        lines = f.readlines()
    
    move_seq, numa_seq = [], []
    for line in lines:
        if (line.startswith('Step') or line.startswith('Sorted Step') or line.startswith('[Break Cycle') or line.startswith('[Undo ')) and ('can move from' in line):
            if line.startswith('Sorted Step 1:'):
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


def resort_and_reindex_moveseq(data_path, logger_path, logger=None):
    def _write_info(info):
        if logger is not None:
            logger.write((info+'\n'))
            logger.flush()
        else:
            print(info)
    
    ecs_env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    ecs_env.reset()
    
    move_seq, numa_seq = parse_logger(logger_path=logger_path)
    log_info = f"Start Resort and Reindex, Num Move {len(move_seq)}!"
    _write_info(log_info)
    
    moves_graph = nx.MultiDiGraph()
    moves_in = (np.array(move_seq)[:, 1]).astype(int)
    moves_out = (np.array(move_seq)[:, 2]).astype(int)
    nodes = list(set(moves_in).union(set(moves_out)))
    edges = list(zip(moves_out, moves_in, range(len(move_seq))))
    moves_graph.add_nodes_from(nodes)
    moves_graph.add_edges_from(edges)
    
    pos_idxes = np.where(np.array(move_seq)[:, -1] > 0)[0]
    pos_rewards = list(zip(pos_idxes, np.array(move_seq)[pos_idxes, -1]))
    pos_rewards = sorted(pos_rewards, key=lambda x: x[1], reverse=True)
    
    move_flag = np.zeros(len(move_seq), dtype=bool)
    move_seq_chunk = []
    for pi, mi in enumerate(pos_rewards):
        mi, ri = mi[0], mi[1]
        if move_flag[mi]:
            continue
        move_idxes = [[mi]]
        layers = [[move_seq[mi][1]]]
        visited = {tuple(es[:2]): False for es in edges}
        while len(layers[-1]) > 0:
            new_layer, new_idxes = [], [[] for _ in range(len(layers[-1]))]
            for li, l in enumerate(layers[-1]):
                for e in moves_graph.adj[l]:
                    if visited[(l, e)]:
                        continue
                    new_layer.append(e)
                    new_idxes[li].extend(list(moves_graph.get_edge_data(l, e).keys()))
                    visited[(l, e)] = True
            move_idxes.append(new_idxes)
            layers.append(new_layer)
        env_bk = dcp(ecs_env)
        this_move_seq = [mi]
        if len(numa_seq[mi]) > 0:
            numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
            numa_action[numa_seq[mi]] = 1
        else:
            numa_action = None
        _, reward, _, infos = ecs_env.step_ignore_resource_satisfaction(list(move_seq[mi][:2]), numa_action)
        move_flag[mi] = True
        unsatisfied_boxes = infos['unsatisfied_boxes']
        di = 1
        while len(unsatisfied_boxes) > 0 and (di < len(move_idxes)):
            # if pi == 17:
            #     pdb.set_trace()
            for b in unsatisfied_boxes:
                bi = 0
                while (bi < len(layers[di-1])) and (b != layers[di-1][bi]): bi += 1;
                if (bi >= len(layers[di-1])):
                    continue
                ms = np.array(move_idxes[di][bi], dtype=int)
                mf = move_flag[ms]
                undo_ms = list(ms[~mf])
                for m in ms[mf]:
                    reward += move_seq[m][-1]
                    this_move_seq = [m] + this_move_seq
                for m in undo_ms:
                    if len(numa_seq[m]) > 0:
                        numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                        numa_action[numa_seq[m]] = 1
                    else:
                        numa_action = None
                    _, r, _, infos = ecs_env.step_ignore_resource_satisfaction(list(move_seq[m][:2]), numa_action)
                    new_boxes = infos['unsatisfied_boxes']
                    reward += r
                    this_move_seq = [m] + this_move_seq
                    move_flag[m] = True
                    if b not in new_boxes:
                        break
            di += 1
            unsatisfied_boxes = infos['unsatisfied_boxes']
        # if len(unsatisfied_boxes) > 0:
        #     pdb.set_trace()
        real_reward, iter_t, this_real_move_seq = 0, 0, []
        this_move_flag = np.zeros(len(this_move_seq), dtype=bool)
        while (not this_move_flag.all()) and (iter_t < len(this_move_seq)):
            iter_t += 1
            for ii, m in enumerate(this_move_seq):
                if this_move_flag[ii]:
                    continue
                if len(numa_seq[m]) > 0:
                    numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                    numa_action[numa_seq[m]] = 1
                else:
                    numa_action = None
                _, rew, _, infos = env_bk.step(list(move_seq[m][:2]), numa_action)
                if infos['action_available']:
                    real_reward += rew
                    this_move_flag[ii] = True
                    this_real_move_seq.append(m)
        if (not this_move_flag.all()):
            real_reward = reward - len(this_move_seq)
            this_real_move_seq = this_move_seq
        log_info = f"Check Chunk {pi}-{mi}, Num Unsatisfied {len(unsatisfied_boxes)}, Num Move {len(this_real_move_seq)}, Total Reward {real_reward}!"
        _write_info(log_info)
        move_seq_chunk.append([this_real_move_seq, -real_reward, -len(this_real_move_seq)])
    
    move_seq_chunk = sorted(move_seq_chunk, key=lambda x: (x[1], x[2]))
    undo_moves = np.where(move_flag==False)[0]
    move_seq_chunk.append([undo_moves, 0, len(undo_moves)])
    log_info = f"Num Undo Moves {len(undo_moves)} After Sorting!"
    _write_info(log_info)
    
    move_flag = np.zeros(len(move_seq), dtype=bool)
    resort_move_seq, reward_list = [], []
    ecs_env.reset()
    iter = 0
    while True:
        undo_moves_list = []
        for ms in move_seq_chunk:
            for m in ms[0]:
                if move_flag[m]:
                    continue
                if len(numa_seq[m]) > 0:
                    numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                    numa_action[numa_seq[m]] = 1
                else:
                    numa_action = None
                _, reward, done, info = ecs_env.step(list(move_seq[m][:2]), numa_action)
                if not info['action_available']:
                    log_info = f"Iter {iter}, Sorted Moveseq Error! Error Info {info['action_info']}"
                    _write_info(log_info)
                    undo_moves_list.append(m)
                    # pdb.set_trace()
                    continue
                log_info = f"Iter {iter}, Step {info['move_count']-info['invalid_action_count']}: {reward}, {done}, {info['action_info']}"
                _write_info(log_info)
                move_flag[m] = True
                resort_move_seq.append([list(move_seq[m][:2]), numa_action])
                reward_list.append(reward)
            
        new_undo = []
        for m in undo_moves_list:
            if move_flag[m]:
                continue
            if len(numa_seq[m]) > 0:
                numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                numa_action[numa_seq[m]] = 1
            else:
                numa_action = None
            _, reward, done, info = ecs_env.step(list(move_seq[m][:2]), numa_action)
            if not info['action_available']:
                # print(info['action_info'])
                new_undo.append(m)
            else:
                log_info = f"Iter {iter}, Step {info['move_count']-info['invalid_action_count']}: {reward}, {done}, {info['action_info']}"
                _write_info(log_info)
                move_flag[m] = True
                resort_move_seq.append([list(move_seq[m][:2]), numa_action])
                reward_list.append(reward)
        undo_moves_list = new_undo
        # pdb.set_trace()
        iter += 1
        if (len(undo_moves_list) <= 0) or ((np.array(move_seq)[undo_moves_list, -1] <= 0).all()):
            break
        if (iter > len(undo_moves)):
            resort_move_seq, reward_list = [], []
            for mi in range(len(move_seq)):
                if len(numa_seq[mi]) > 0:
                    numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                    numa_action[numa_seq[mi]] = 1
                else:
                    numa_action = None
                resort_move_seq.append([list(move_seq[mi][:2]), numa_action])
                reward_list.append(move_seq[mi][-1])
            break
    
    neg_i = 0
    for ri in reward_list[::-1]:
        if ri <= 0:
            neg_i += 1
        else:
            break
    resort_move_seq = resort_move_seq[:len(resort_move_seq)-neg_i]
    
    dense_ecs_env = ECSRawDenseEnvironment(
        data_path=data_path,
        is_limited_count=True, is_filter_unmovable=True,
        is_dense_items=True, is_dense_boxes=False,
        is_state_merge_placement=False, is_normalize_state=True, is_process_numa=True,
    )
    dense_ecs_env.reset()
    
    # unfilter_to_filter_map = dense_ecs_env.ecs_env.init_idxes_map
    dense_items_map, dense_items_idxes = dense_ecs_env.dense_items_map, dense_ecs_env.dense_items_idxes
    dense_items_map = {k: sorted(v) for k, v in dense_items_map.items()}
    dense_items_used = {k: np.zeros(len(v)) for k, v in dense_items_map.items()}
    reindex_move_seq = []
    for ms in resort_move_seq:
        m, numa = ms[0], ms[1]
        item_i = m[0]
        dense_idx = dense_items_idxes[item_i]
        idx = np.where(dense_items_used[dense_idx] == 0)[0][0]
        reindex_move_seq.append([[dense_items_map[dense_idx][idx], m[1]], numa])
        dense_items_used[dense_idx][idx] = 1
    
    ecs_env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    ecs_env.reset()
    for m in reindex_move_seq:
        _, reward, done, info = ecs_env.step(m[0], m[1])
        if not info['action_available']:
            log_info = f"Resort and Reindex Error! Error Info {info['move_count']}: {info['action_info']}"
            reindex_move_seq = []
            for mi in range(len(move_seq)):
                a = [int(move_seq[mi][0]), int(move_seq[mi][1])]
                if len(numa_seq[mi]) > 0:
                    numa_action = np.zeros(ecs_env.max_box_numa, dtype=int)
                    numa_action[numa_seq[mi]] = 1
                else:
                    numa_action = None
                reindex_move_seq.append([a, numa_action])
            break
    
    # ecs_env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    ecs_env.reset()
    init_score = ecs_env.get_cur_state_score()
    init_vio_cost, init_vio_info = ecs_env.get_cur_state_vio_cost()
    log_info = f"Resort and Reindex {data_path}, Init Score {init_score}!\n"
    log_info += f"Initial State Vio Cost: {init_vio_cost}, {init_vio_info}"
    _write_info(log_info)
    total_reward = 0
    for m in reindex_move_seq:
        _, reward, done, info = ecs_env.step(m[0], m[1])
        log_info = f"[Resort and Reindex] Step {info['move_count']}: {reward}, {done}, {info['action_info']}\n"
        log_info += f"[Resort and Reindex] SVio {info['move_count']}: {info['vio_info']}"
        _write_info(log_info)
        total_reward += reward
    final_score = ecs_env.get_cur_state_score()
    vio_cost, vio_info = ecs_env.get_cur_state_vio_cost()
    log_info = f"Resort and Reindex Finished! Final Score {final_score}, Total Reward {total_reward}, Num Move {len(reindex_move_seq)}!\n"
    log_info += f"Final Vio Cost: {vio_cost}, {vio_info}"
    _write_info(log_info)


if __name__ == '__main__':
    data_type = 'ecs_data'
    logger_root_path = f"../results/ecs_opt/opt_logs/{data_type}_addmig1.0_move_seq"
    logger_save_path = f'../results/ecs_opt/resort_logs/{data_type}_addmig1.0_resort_and_reindex'
    os.makedirs(logger_save_path, exist_ok=True)
    
    def _task(data_name):
        data_name = data_name.split('.log')[0]
        data_path = f'../data/{data_type}/{data_name}'
        logger_path = f'{logger_root_path}/{data_name}.log'
        logger = open(f"{logger_save_path}/{data_name}.log", 'w')
        resort_and_reindex_moveseq(
            data_path=data_path,
            logger_path=logger_path,
            logger=logger,
        )
    
    pool = Pool(processes=4)
    for data_name in os.listdir(logger_root_path):
        pool.apply_async(_task, args=(data_name, ))
    pool.close()
    pool.join()
