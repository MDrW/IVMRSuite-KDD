import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from scipy import sparse
import json
import time
from ecs_env.ecs_base_env import ECSEnvironment
import networkx as nx
from copy import deepcopy as dcp
from argparse import ArgumentParser
from multiprocessing import Pool
import pdb


class Placement2Moveseq:
    def __init__(self, args):
        self.args = args
        
        self.data_path = os.path.join(args.data_root_path, args.data_name)
        self.sol_save_path = os.path.join(args.sol_save_root_path, f"{args.data_name}_{args.data_suffix}_placement.npz")
        self.numa_save_path = os.path.join(args.sol_save_root_path, f"{args.data_name}_{args.data_suffix}_numa.npz")
        
        if os.path.exists(args.logger_root_path) and os.path.exists(self.sol_save_path) and os.path.exists(self.numa_save_path):
            self.logger = open(os.path.join(args.logger_root_path, args.data_name+'.log'), 'w')
        else:
            self.logger = None
    
    def _write_info(self, info):
        if self.logger is not None:
            self.logger.write((info+'\n'))
            self.logger.flush()
        else:
            print(info)
    
    def placement_to_moveseq(self):
        if (not os.path.exists(self.sol_save_path)) or (not os.path.exists(self.numa_save_path)):
            return [], []
        st = time.time()
        self.ecs_env = ECSEnvironment(data_path=self.data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
        self.init_placement = self.ecs_env.cur_placement.copy()
        self.item_count, self.box_count = len(self.ecs_env.item_cur_state), len(self.ecs_env.box_cur_state)
        opt_sol = sparse.load_npz(self.sol_save_path).toarray()
        opt_numa_sol = sparse.load_npz(self.numa_save_path).toarray()
        opt_numa_sol = opt_numa_sol.reshape(len(opt_numa_sol), self.box_count, self.ecs_env.max_box_numa)
        opt_numa_sol[opt_numa_sol > 0] = 1
        
        init_score = self.ecs_env.get_cur_state_score()
        self.ecs_env.cur_placement = opt_sol
        opt_score = self.ecs_env.get_cur_state_score()
        self.ecs_env.cur_placement = np.array(self.init_placement).copy()
        st1 = time.time()
    
        reward_matrix = self.ecs_env._get_reward_batch(range(self.box_count))
        moves, reward, numa_moves = [], [], []
        for i, j in zip(*np.where(opt_sol != self.init_placement)):
            if opt_sol[i, j]:
                moves.append([i, j])
                reward.append(reward_matrix[i, j])
                numa_moves.append(opt_numa_sol[i, j])
        sorted_idxes = np.argsort(reward)[::-1]
        moves = np.array(moves, dtype=int)[sorted_idxes]
        numa_moves = np.array(numa_moves, dtype=int)[sorted_idxes]
        st2 = time.time()
        info = f"Initial Score: {init_score}, Optimal Score: {opt_score}, Optimal Move Count: {len(moves)}! Elapsed Time: {st1-st}s, {st2-st1}s."
        self._write_info(info)
        
        self.real_moves, self.real_numa_moves, self.org_boxes, self.rewards = [], [], [], []
        self.break_cycle_moves = 0
        while len(moves) > 0:
            move_flag, state_change = np.zeros(len(moves), dtype=bool), 0
            for i, m in enumerate(moves):
                if self.ecs_env.get_available_actions()[m[0], m[1]] > 0:
                    st3 = time.time()
                    item_cur_box = self.ecs_env.get_item_cur_box(list(m)[0])
                    self.org_boxes.append(item_cur_box)
                    _, reward, done, infos = self.ecs_env.step(list(m), numa_moves[i])
                    self.rewards.append(reward)
                    self.real_moves.append(list(m))
                    self.real_numa_moves.append(infos['numa_action'])
                    if self.break_cycle_moves > 0:
                        self.break_cycle_moves += 1
                    move_count, action_info = infos["move_count"], infos["action_info"]
                    st4 = time.time()
                    info = f"Step {move_count}: {reward}, {done}, {action_info}, {st4-st3}s."
                    self._write_info(info)
                    
                    if done:
                        state_change += 1
                        # break
                    else:
                        state_change += 1
                else:
                    move_flag[i] = True
                
            moves = moves[move_flag]
            numa_moves = numa_moves[move_flag]
            if (state_change == 0):
                if self.args.is_solve_cycle:
                    _, undo_moves, undo_numa_moves = self._undo_neg_moves()
                    moves = np.array(list(moves) + list(undo_moves))
                    numa_moves = np.array(list(numa_moves) + list(undo_numa_moves))
                    moves, numa_moves, state_change = self._solve_cycle(moves, numa_moves)
                if state_change == 0:
                    break
        
        if self.args.is_undo_neg:
            self._undo_neg_moves()
            self.try_to_undo_moves()
        
        # self.real_numa_moves = dcp(self.ecs_env.stored_numa_actions)
        
        final_score_1 = self.ecs_env.get_cur_state_score()
        total_reward, costs, invalid_num = self.ecs_env.eval_move_sequence(self.real_moves, self.real_numa_moves)
        final_score_2 = self.ecs_env.get_cur_state_score()
        
        st5 = time.time()
        info = f"Real Move Count: {len(self.real_moves)}, Numa Count: {len(self.real_numa_moves)}, Invalid Num: {invalid_num}, Total Reward: {np.sum(self.rewards)}, Final Score: {final_score_1}, Eval Reward: {total_reward}, Eval Final Score: {final_score_2}, Elapsed Time: {st5-st}s."
        self._write_info(info)
        
        if self.args.is_sorted_moves and (len(self.real_moves) > 0):
            tr, fs = self.sort_moves()
            total_reward, costs, invalid_num = self.ecs_env.eval_move_sequence(self.real_moves, self.real_numa_moves)
            st6 = time.time()
            info = f"After Sorting Moves, Move Count: {len(self.real_moves)}, Invalid Num: {invalid_num}, Total Reward: {tr}, Final Score: {fs}, Eval Reward: {total_reward}, Elapsed Time: {st6-st}s, WithoutSortedTime {st5-st}s."
            self._write_info(info)
        
        return self.real_moves, self.real_numa_moves
    
    def try_to_undo_moves(self):
        self._write_info(f"Start Trying to Undo Moves...")
        move_flag = np.ones(len(self.real_moves), dtype=bool)
        can_undo = np.ones(len(self.real_moves), dtype=int) * -1
        pos_rewards = []
        simu_env = dcp(self.ecs_env)
        for i, r in enumerate(self.rewards):
            if r > 0:
                pos_rewards.append(i)
                action = [*self.real_moves[i], self.org_boxes[i]]
                undo_flag = simu_env.is_undo_action_available(action)[0]
                if undo_flag:
                    can_undo[i] = i
            else:
                action = [*self.real_moves[i], self.org_boxes[i]]
                if simu_env.is_undo_action_available(action)[0]:
                    can_undo[i] = -10
                    simu_env.undo_step(action)
        
        state_change = True
        pos_undo_flag = np.zeros(len(pos_rewards), dtype=bool)
        total_get_rewards, total_undo_num = 0, 0
        while state_change:
            state_change = False
            for pi, mi in enumerate(pos_rewards[::-1]):
                if (can_undo[mi] == -1) or pos_undo_flag[pi]:
                    continue
                simu_env = dcp(self.ecs_env)
                undo_flag = np.zeros(len(self.real_moves), dtype=bool)
                undo_i, undo_rewards, log_info = 0, 0, ''
                change_can_undo = []
                mi = can_undo[mi]
                for j in range(mi, -1, -1):
                    if (j != mi) and (can_undo[j] >= 0):
                        continue
                    action = [*self.real_moves[j], self.org_boxes[j]]
                    
                    if simu_env.is_undo_action_available(action)[0]:
                        if (j != mi) and (self.rewards[j] > 0):
                            change_can_undo.append(j)
                            continue
                        
                        _, reward, done, info = simu_env.undo_step(action)
                        log_info += f"[Undo Seq] Step {pi}-{undo_i}[{j}]: {reward}, {done}, {info['action_info']}\n"
                        undo_flag[j] = True
                        undo_rewards += reward
                        undo_i += 1
                
                if (undo_i > 0) and (undo_rewards >= 0.):
                    check_env = dcp(simu_env)
                    check_env.reset()
                    for j in range(len(self.real_moves)):
                        if (~move_flag | undo_flag)[j]:
                            continue
                        _, _, d, info = check_env.step(list(self.real_moves[j]), self.real_numa_moves[j])
                        if (not info['action_available']) or ((j != len(self.real_moves) - 1) and d):
                            for k in range(j):
                                if can_undo[k] == -10 and undo_flag[k] and move_flag[k]:
                                    _, r, d, info = check_env.step(list(self.real_moves[k]), self.real_numa_moves[k])
                                    undo_i -= 1
                                    undo_rewards -= r
                                    undo_flag[k] = False
                            _, _, d, info = check_env.step(list(self.real_moves[j]), self.real_numa_moves[j])
                            if (not info['action_available']) or ((j != len(self.real_moves) - 1) and d):
                                undo_i = 0
                                break
                
                for ci in change_can_undo:
                    if (undo_i > 0) and (undo_rewards >= 0.):
                        can_undo[ci] = ci
                    else:
                        can_undo[ci] = mi
                
                if (undo_i > 0) and (undo_rewards >= 0.):
                    self.ecs_env = dcp(simu_env)
                    state_change = True
                    pos_undo_flag[pi] = True
                    total_get_rewards += undo_rewards
                    total_undo_num += undo_i
                    move_flag = move_flag & (~undo_flag)
                    log_info += f"Undo Seq {pi}, Num Undo {undo_i}, Get Total Reward {undo_rewards}!"
                    self._write_info(log_info)
        
        log_info =  f"Undo Seq Total Undo Num: {total_undo_num}, Get Reward: {total_get_rewards}!"
        self._write_info(log_info)
        
        self.real_moves = np.array(self.real_moves)[move_flag].tolist()
        self.real_numa_moves = np.array(self.real_numa_moves)[move_flag].tolist()
        self.org_boxes = np.array(self.org_boxes)[move_flag].tolist()
        self.rewards = np.array(self.rewards)[move_flag].tolist()
    
    def sort_moves(self):
        move_seq = list(map(list, zip(
            np.array(self.real_moves)[:, 0].tolist(), 
            np.array(self.real_moves)[:, 1].tolist(), 
            list(self.org_boxes), 
            list(self.rewards)
        )))
        
        sorted_move_seq, self.real_numa_moves = self._sort_move_seq(move_seq)
        self.real_moves = np.array(sorted_move_seq)[:, :2].astype(int).tolist()
        self.org_boxes = np.array(sorted_move_seq)[:, 2].astype(int).tolist()
        self.rewards = np.array(sorted_move_seq)[:, 3].astype(float).tolist()
        
        # self.ecs_env.reset()
        ecs_env = ECSEnvironment(data_path=self.data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
        total_reward = 0
        for m, nm in zip(self.real_moves, self.real_numa_moves):
            _, r, d, info = ecs_env.step(list(m), nm)
            total_reward += r
            info = f"Sorted Step {info['move_count']}: {r}, {d}, {info['action_info']}"
            self._write_info(info)
        final_score = ecs_env.get_cur_state_score()
        
        return total_reward, final_score
    
    def _sort_move_seq(self, move_seq):
        self._write_info(f"Start Sorting Move Sequence...")
        ecs_env = ECSEnvironment(data_path=self.data_path, is_limited_count=False, is_filter_unmovable=True, is_process_numa=True)
        ecs_env.reset()
        
        moves_graph = nx.MultiDiGraph()
        moves_in = np.array(move_seq)[:, 1]
        moves_out = np.array(move_seq)[:, 2]
        nodes = list(set(moves_in).union(set(moves_out)))
        edges = list(zip(moves_out, moves_in, range(len(move_seq))))
        moves_graph.add_nodes_from(nodes)
        moves_graph.add_edges_from(edges)
        
        pos_rewards, stored_rewards = [], []
        for mi in range(len(move_seq)):
            action = list(move_seq[mi][:2])
            _, reward, _, info = ecs_env.step(action, self.real_numa_moves[mi])
            if reward > 0:
                pos_rewards.append([mi, reward])
            stored_rewards.append(reward)
        pos_rewards = sorted(pos_rewards, key=lambda x: x[1], reverse=True)
        
        def _sort_indicator(target_r, real_r):
            if real_r > target_r:
                gap = (real_r - target_r) / target_r
                if gap >= 0.5:
                    return 0
                else:
                    return 1
            else:
                gap = (target_r - real_r) / target_r
                if gap < 0.1:
                    return 2
                else:
                    return 3
        move_seq_chunk = []
        move_flag = np.zeros(len(move_seq), dtype=bool)
        item_chunk_idxes = {item_i: [0, -1, -1] for item_i in np.array(move_seq)[:, 0]}
        ecs_env.reset()
        for mi in pos_rewards:
            mi, ri = mi[0], mi[1]
            if move_flag[mi]:
                continue
            action = move_seq[mi][:2]
            move_idxes = []
            for edge in nx.subgraph(moves_graph, nx.bfs_tree(moves_graph, move_seq[mi][1]).nodes).edges():
                move_idxes.extend(list(moves_graph.get_edge_data(*edge).keys()))
            move_idxes.append(mi)
            move_idxes = sorted(set(move_idxes))
            
            total_reward, ind_reward, flag = 0, 0, True
            this_chunk = []
            copy_chunk_idxes = dcp(item_chunk_idxes)
            for idx in move_idxes:
                if move_flag[idx]:
                    this_chunk.append(idx)
                    total_reward += stored_rewards[idx]
                    copy_chunk_idxes[move_seq[idx][0]][0] += 1
                    continue
                
                _, reward, _, info = ecs_env.step(list(move_seq[idx][:2]), self.real_numa_moves[idx])
                if not info['action_available']:
                    # pdb.set_trace()
                    flag = False
                    self._write_info(f"Check Chunk {mi} Error In Step {idx}! {info['action_info']}")
                    return move_seq, self.real_numa_moves
                total_reward += reward
                ind_reward += reward
                stored_rewards[idx] = reward
                copy_chunk_idxes[move_seq[idx][0]] = [1, len(move_seq_chunk), reward]
                this_chunk.append(idx)
            
            if flag:
                move_flag[move_idxes] = True
                item_chunk_idxes = copy_chunk_idxes
                move_seq_chunk.append([this_chunk, -total_reward, len(this_chunk), -ri, _sort_indicator(ri, ind_reward), -ind_reward])
                self._write_info(f"Check Chunk {mi} Finished! Influence Num {len(this_chunk)} Moves, Total Reward {total_reward}!")
        
        for _, key in item_chunk_idxes.items():
            if key[0] > 1:
                move_seq_chunk[key[1]][5] += key[-1]
                move_seq_chunk[key[1]][4] = _sort_indicator(-move_seq_chunk[key[1]][3], -move_seq_chunk[key[1]][5])
        # pdb.set_trace()
        move_seq_chunk = sorted(move_seq_chunk, key=lambda x: (x[3], x[5], x[4], x[1], x[2]))
        uncheck_rew = np.array(move_seq)[:, 3][~move_flag].sum()
        move_seq_chunk.append([np.arange(len(move_seq))[~move_flag].tolist(), -uncheck_rew, (~move_flag).sum(), -uncheck_rew, 4, -uncheck_rew])
        sorted_move_seq, sorted_numa_moves = [], []
        move_idx, change_num = 0, 0
        move_flag = np.zeros(len(move_seq), dtype=bool)
        changed_idxes = np.ones(len(move_seq), dtype=int) * -1
        try_undo_moves_list = []
        for moves in move_seq_chunk:
            action_list, this_num = [], 0
            for idx in moves[0]:
                if move_flag[idx]:
                    action_list.append(changed_idxes[idx])
                    continue
                else:
                    sorted_move_seq.append(move_seq[idx])
                    sorted_numa_moves.append(self.real_numa_moves[idx])
                    changed_idxes[idx] = move_idx
                    action_list.append(move_idx)
                    this_num += 1
                    if idx != move_idx:
                        change_num += 1
                    move_idx += 1
                    move_flag[idx] = True
            if (moves[-1] >= 0.) and (this_num > 0):
                try_undo_moves_list.append(action_list)
                # pdb.set_trace()
        # pdb.set_trace()
        if len(move_seq) != len(sorted_move_seq):
            self._write_info(f"Original Move Seq {len(move_seq)} is not equal to Sorted Move Seq {len(sorted_move_seq)}! Exit Sorted!")
            return move_seq, self.real_numa_moves
        self._write_info(f"Finish Sorted Move Sequence! Check Seq Chunk Number: {len(move_seq_chunk)}! Total Move Num {len(sorted_move_seq)}, Not Check Number {(~move_flag).sum()}, Change Location Number by Sorted {change_num}!")
        
        return sorted_move_seq, sorted_numa_moves
    
    def _solve_cycle_by_box(self, moves, numa_moves, move_flag, undo_idxes, target_idx, 
                            simu_env, nodes, reward_matrix):
        candidate_moves, nodes_importance, imp = [], [], len(nodes)
        for n in nodes:
            item_idxes_in_box = np.where(simu_env.cur_placement[:, n] > 0)[0]
            movable_idxes = np.where(simu_env.actions_available[item_idxes_in_box] > 0)
            candidate_moves.extend(zip(item_idxes_in_box[movable_idxes[0]], movable_idxes[1]))
            nodes_importance.extend([imp] * len(movable_idxes[0]))
            imp -= 1
        if len(candidate_moves) <= 0:
            return False, simu_env, reward_matrix, move_flag, 0
        candidate_moves = np.array(candidate_moves)
        to_box_in_cycle = [0 if cm[1] in nodes else 1 for cm in candidate_moves]
        candidate_moves = candidate_moves[np.lexsort((reward_matrix[candidate_moves[:, 0], candidate_moves[:, 1]], nodes_importance, to_box_in_cycle))[::-1]]
        # pdb.set_trace()
        new_moves, new_numa_moves, cidx = [], [], 0
        copy_origin_env = dcp(simu_env)
        if len(candidate_moves) > 0:
            copy_simu_env = dcp(simu_env)
            this_move_flag = np.ones(len(moves), dtype=bool)
            is_new_move = []
            while this_move_flag[target_idx]:
                num_move = 0
                for ci, i in enumerate(undo_idxes):
                    if (not move_flag[i]) and this_move_flag[i] and (simu_env.get_available_actions()[moves[i][0], moves[i][1]] > 0):
                        _, _, _, sinfos = simu_env.step(list(moves[i]), numa_moves[i])
                        num_move += 1
                        this_move_flag[i] = False
                        new_moves.append(list(moves[i]))
                        new_numa_moves.append(sinfos['numa_action'])
                        is_new_move.append(False)
                if num_move <= 0:
                    while (cidx < len(candidate_moves)) and (simu_env.get_available_actions()[candidate_moves[cidx][0], candidate_moves[cidx][1]] <= 0):
                        cidx += 1
                    if cidx >= len(candidate_moves):
                        break
                    _, _, _, sinfos = simu_env.step(list(candidate_moves[cidx]))
                    new_moves.append(list(candidate_moves[cidx]))
                    new_numa_moves.append(sinfos['numa_action'])
                    is_new_move.append(True)
                    cidx += 1
                    num_move += 1
                if num_move <= 0:
                    break
                # pdb.set_trace()
            if this_move_flag[target_idx]:
                simu_env = copy_simu_env
                new_moves = []
                new_numa_moves = []
        is_solved, break_step = False, 0
        if len(new_moves) > 0:
            available = True
            full_rewards, org_boxes = [], []
            copy_reward_matrix = dcp(reward_matrix)
            for i, m in enumerate(new_moves):
                item_cur_box = self.ecs_env.get_item_cur_box(m[0])
                org_boxes.append(item_cur_box)
                _, reward, done, info = self.ecs_env.step(list(m), new_numa_moves[i])
                log_info = f"[Break Cycle by Box] Step {break_step} {'New' if is_new_move[i] else 'Origin'}: {reward}, {done}, {info['action_info']}"
                self._write_info(log_info)
                break_step += 1
                full_rewards.append(reward)
                available_actions = self.ecs_env.get_available_actions()
                box_update = [int(item_cur_box), m[1]]
                reward_update = self.ecs_env._get_reward_batch(box_update)
                reward_matrix[:, box_update] = reward_update
                reward_matrix[(1-available_actions).astype(bool)] = -1e99
                if not info['action_available']:
                    for j in range(i-1, -1, -1):
                        self.ecs_env.undo_step(list(new_moves[j]))
                    available = False
                    reward_matrix = copy_reward_matrix
                    break
            if available:
                is_solved = True
                self.real_moves.extend(new_moves)
                self.real_numa_moves.extend(new_numa_moves)
                self.rewards.extend(full_rewards)
                self.org_boxes.extend(org_boxes)
                move_flag = move_flag | (~this_move_flag)
                info = f"Have Broken Move {target_idx} by Boxes, Move Count {len(new_moves)}, Get Reward {np.sum(full_rewards)}!"
            else:
                is_solved = False
                simu_env = copy_origin_env
                info = f"Got Move Sequence is not available! Breaking Move {target_idx} by Boxes has Failed!"
        else:
            is_solved = False
            simu_env = copy_origin_env
            info = f"Breaking Move {target_idx} by Boxes has Failed! Num Undo: {len(undo_idxes)}."
        self._write_info(info)
        
        return is_solved, simu_env, reward_matrix, move_flag, break_step
    
    def _solve_cycle_by_item(self, moves, numa_moves, move_flag, undo_idxes, target_idx, 
                             simu_env, reward_matrix, stored_rewards, out_degrees, in_degrees, reserved_boxes):
        def _try_break_cycle(mi, mr, break_step, is_check_rew=True):
            if (simu_env.actions_available[moves[mi][0]].sum() > 0):
                if is_check_rew:
                    max_r = np.max(reward_matrix[moves[mi][0]])
                    if max_r < mr and (np.abs(mr - max_r) / (min(abs(mr), abs(max_r)) + 1) > 10):
                        return False, None, None, 0., None
                is_box_reserved = np.zeros(len(reward_matrix[0]), dtype=bool)
                is_box_reserved[reserved_boxes] = True
                if (simu_env.actions_available[moves[mi][0], ~is_box_reserved].sum() <=0):
                    is_box_reserved = np.zeros(len(reward_matrix[0]), dtype=bool)
                item_cur_box = simu_env.get_item_cur_box(moves[mi][0])
                # pdb.set_trace()
                to_box = np.lexsort((out_degrees, in_degrees, -reward_matrix[moves[mi][0]], is_box_reserved.astype(int)))[0]
                action = [moves[mi][0], to_box]
                _, reward, done, infos = simu_env.step(action)
                info = f"[Break Cycle by Item] Step {break_step} try to Break {mi}{moves[mi]}: {reward}, {done}, {infos['action_info']}"
                self._write_info(info)
                
                available_actions = simu_env.get_available_actions()
                box_update = [int(item_cur_box), action[1]]
                reward_update = simu_env._get_reward_batch(box_update)
                reward_matrix[:, box_update] = reward_update
                reward_matrix[(1-available_actions).astype(bool)] = -1e99
                
                # moves_graph.remove_edge(item_cur_box, moves[mi][1])
                # moves_graph.add_edge(item_cur_box, action[1])
                in_degrees[moves[mi][1]] -= 1
                in_degrees[action[1]] += 1
                return True, action, infos['numa_action'], reward, item_cur_box
            else:
                return False, None, None, 0., None
        
        def _step(mi, break_step):
            action = list(moves[mi])
            if simu_env.actions_available[action[0], action[1]]:
                item_cur_box = simu_env.get_item_cur_box(action[0])
                _, reward, done, infos = simu_env.step(action, numa_moves[mi])
                info = f"[Break Cycle by Item] Step {break_step} step to Break {mi}{moves[mi]}: {reward}, {done}, {infos['action_info']}"
                self._write_info(info)
                
                available_actions = simu_env.get_available_actions()
                box_update = [int(item_cur_box), action[1]]
                reward_update = simu_env._get_reward_batch(box_update)
                reward_matrix[:, box_update] = reward_update
                reward_matrix[(1-available_actions).astype(bool)] = -1e99
                return True, action, infos['numa_action'], reward, item_cur_box
            else:
                return False, None, None, 0., None
        
        def _first_step_then_try(mi, mr, break_step, is_check_rew=True):
            is_movable, action, numa_action, reward, item_cur_box = _step(mi, break_step)
            if (not is_movable):
                is_movable, action, numa_action, reward, item_cur_box = _try_break_cycle(mi, mr, break_step, is_check_rew)
            return is_movable, action, numa_action, reward, item_cur_box
        
        copy_env, copy_reward_matrix = dcp(simu_env), dcp(reward_matrix)
        is_try = False
        break_step = 0
        step_moves, step_numa_moves, full_rewards, org_boxes = [], [], [], []
        this_move_flag = np.zeros(len(moves), dtype=bool)
        while target_idx in undo_idxes:
            state_change = False
            new_undo, cur_i = [], 0
            for idx in undo_idxes[::-1]:
                cur_i += 1
                if move_flag[idx]:
                    continue
                is_movable, action, numa_action, reward, cur_box = _first_step_then_try(idx, stored_rewards[idx], break_step, is_check_rew=True)
                if is_movable:
                    break_step += 1
                    step_moves.append(action)
                    step_numa_moves.append(numa_action)
                    full_rewards.append(reward)
                    org_boxes.append(cur_box)
                    this_move_flag[idx] = True
                    state_change = True
                    is_try=False
                    # new_undo.extend(undo_idxes[cur_i:])
                    # break
                else:
                    new_undo.append(idx)
            undo_idxes = new_undo[::-1]
            if (not state_change) and (target_idx in undo_idxes):
                # break
                if not is_try:
                    is_try = True
                    to_boxes = [moves[idx][1] for idx in undo_idxes]
                    from_boxes = [simu_env.get_item_cur_box(moves[idx][0]) for idx in undo_idxes]
                    graph = nx.MultiDiGraph()
                    nodes = list(set(to_boxes).union(set(from_boxes)))
                    edges = list(zip(from_boxes, to_boxes, range(len(from_boxes))))
                    graph.add_nodes_from(nodes)
                    graph.add_edges_from(edges)
                    new_idxes = [0]
                    for edge in nx.subgraph(graph, nx.bfs_tree(graph, moves[target_idx][1]).nodes).edges():
                        new_idxes.extend(list(graph.get_edge_data(*edge).keys()))
                    new_idxes = sorted(list(set(new_idxes)))
                    undo_idxes = np.array(undo_idxes)[new_idxes].tolist()
                else:
                    break
        if (target_idx in undo_idxes):
            simu_env, reward_matrix = copy_env, copy_reward_matrix
            break_step, step_moves, step_numa_moves, full_rewards, org_boxes = 0, [], [], [], []
            this_move_flag = np.zeros(len(moves), dtype=bool)
        # pdb.set_trace()
        is_solved = False
        if len(step_moves) > 0:
            available = True
            for i, m in enumerate(step_moves):
                _, _, _, info = self.ecs_env.step(list(m), step_numa_moves[i])
                if not info['action_available']:
                    for j in range(i-1, -1, -1):
                        self.ecs_env.undo_step(list(step_moves[j]))
                    available = False
                    break
            if available:
                is_solved = True
                self.real_moves.extend(step_moves)
                self.real_numa_moves.extend(step_numa_moves)
                self.rewards.extend(full_rewards)
                self.org_boxes.extend(org_boxes)
                move_flag = move_flag | this_move_flag
                info = f"Have Broken Move {target_idx} by Items, Move Count {len(step_moves)}, Get Reward {np.sum(full_rewards)}!"
            else:
                is_solved = False
                simu_env = copy_env
                info = f"Got Move Sequence is not available! Breaking Move {target_idx} by Items has Failed!"
        else:
            is_solved = False
            simu_env = copy_env
            info = f"Breaking Move {target_idx} by Items has Failed! Num Undo: {len(undo_idxes)}."
        self._write_info(info)
        
        return is_solved, simu_env, reward_matrix, move_flag, break_step
    
    def _solve_cycle(self, moves, numa_moves):
        moves_graph = nx.MultiDiGraph()
        moves_in = np.array(moves)[:, 1]
        moves_out = np.array([self.ecs_env.item_cur_box[m[0]] for m in moves])
        nodes = list(set(moves_in).union(set(moves_out)))
        edges = list(zip(moves_out, moves_in, range(len(moves))))
        moves_graph.add_nodes_from(nodes)
        moves_graph.add_edges_from(edges)
        
        available_actions = self.ecs_env.get_available_actions()
        reward_matrix = self.ecs_env._get_reward_batch(range(self.box_count))
        pos_rewards, stored_rewards, reserved_boxes = [], [], []
        for mi, m in enumerate(moves):
            if reward_matrix[m[0], m[1]] > 0:
                pos_rewards.append([mi, reward_matrix[m[0], m[1]]])
                reserved_boxes.append(moves[mi][1])
            stored_rewards.append(reward_matrix[m[0], m[1]])
        reward_matrix[(1-available_actions).astype(bool)] = -1e99
        pos_rewards = sorted(pos_rewards, key=lambda x: x[1], reverse=True)
        in_degrees = np.zeros(reward_matrix.shape[1], dtype=int)
        out_degrees = np.zeros(reward_matrix.shape[1], dtype=int)
        for d, n in dict(moves_graph.in_degree()).items():
            in_degrees[d] += n
        for d, n in dict(moves_graph.out_degree()).items():
            out_degrees[d] += n
        move_flag = np.zeros(len(moves), dtype=bool)
        
        simu_env = dcp(self.ecs_env)
        total_num_move = 0
        for pos_m in pos_rewards:
            mi, mr = pos_m
            
            new_idxes = []
            undo_nodes = nx.bfs_tree(moves_graph, moves[mi][1]).nodes
            for edge in nx.subgraph(nx.DiGraph(moves_graph), undo_nodes).edges():
                new_idxes.extend(list(moves_graph.get_edge_data(*edge).keys()))
            undo_idxes = [mi] + new_idxes
            # pdb.set_trace()
            is_solved, simu_env, reward_matrix, move_flag, break_step = self._solve_cycle_by_item(
                moves, numa_moves, move_flag, undo_idxes, mi, 
                simu_env, reward_matrix, stored_rewards, out_degrees, in_degrees, reserved_boxes
            )
            if not is_solved:
                is_solved, simu_env, reward_matrix, move_flag, break_step = self._solve_cycle_by_box(
                    moves, numa_moves, move_flag, undo_idxes, mi, 
                    simu_env, [n for n in undo_nodes], reward_matrix)
            total_num_move += break_step
        
        tr = np.sum(self.rewards[-total_num_move:]) if total_num_move > 0 else 0.
        info = f"Finish Reverse Solving Cycle! Num Change {total_num_move}, Num Undo {np.sum(~move_flag)}, Total Reward {tr}!"
        self._write_info(info)
        return np.array(moves)[~move_flag], np.array(numa_moves)[~move_flag], (np.sum(move_flag) > 0)
    
    def _check_action_can_undo(self, action, env, undo_env, check_action_list, check_numa_action_list):
        undo_flag = True
        simulation_env = dcp(env)
        _, _, _, info = simulation_env.undo_step(action)
        if not info['action_available']:
            undo_flag = False
        else:
            if check_action_list:
                undo_simu_env = dcp(undo_env)
                for ci, ck_act in enumerate(check_action_list):
                    if check_numa_action_list:
                        numa_action = check_numa_action_list[ci]
                    else:
                        numa_action = None
                    _, _, d, info = undo_simu_env.step(ck_act, numa_action)
                    if not info['action_available']:
                        undo_flag = False
                        break
                    # if (ci != len(check_action_list)-1) and d:
                    #     undo_flag = False
                    #     break
        return undo_flag
    
    def _undo_move(self, action, num_undo_move, undo_env=None, check_action_list=None, check_numa_action_list=None):
        st1 = time.time()
        undo_flag = self._check_action_can_undo(action, self.ecs_env, undo_env, check_action_list, check_numa_action_list)
        
        if undo_flag:
            num_undo_move += 1
            _, reward, done, infos = self.ecs_env.undo_step(action)
            action_info = infos["action_info"]
            st2 = time.time()
            info = f"[Undo] Step {num_undo_move}: {reward}, {done}, {action_info}, {st2-st1}s."
            self._write_info(info)
        return undo_flag, num_undo_move
    
    def _update_undo_move_infos(self, move_flag, num_undo_move, undo_type='NegetiveReward'):
        assert (~move_flag).sum() == num_undo_move, f"Num Undo Moves Error: {(~move_flag).sum()} != {num_undo_move}!"
        get_reward = -np.array(self.rewards)[~move_flag].sum()
        self.real_moves = np.array(self.real_moves)[move_flag].tolist()
        self.real_numa_moves = np.array(self.real_numa_moves)[move_flag].tolist()
        self.rewards = np.array(self.rewards)[move_flag].tolist()
        self.org_boxes = np.array(self.org_boxes)[move_flag].tolist()
        self._write_info(f"Undo {undo_type} Moves Finished! Num Undo: {num_undo_move}, Get Reward: {get_reward}!")
    
    def _undo_neg_moves(self, not_check_list=None):
        self._write_info(f"Start Check and Undo NegetiveReward Moves...")
        # sorted_idxes = np.argsort(self.rewards)
        sorted_idxes = np.array(range(len(self.rewards)))[::-1]
        move_flag = np.ones(len(self.real_moves), dtype=bool)
        if not_check_list is not None:
            move_flag[not_check_list] = False
        
        num_undo_move, state_change = 0, True
        while state_change:
            new_undo_num = 0
            undo_env = dcp(self.ecs_env)
            to_boxes_map = {i: [] for i in range(len(self.ecs_env.box_cur_state))}
            for i in sorted_idxes:
                action = [self.real_moves[i][0], self.real_moves[i][1], self.org_boxes[i]]
                undo_env.undo_step(action)
                undo_flag = False
                if (move_flag[i]) and (self.ecs_env.is_undo_action_available(action)[0]):
                    if (self.rewards[i] <= 0):
                        undo_flag, num_undo_move = self._undo_move(
                            action=action, 
                            num_undo_move=num_undo_move,
                            undo_env=undo_env,
                            check_action_list=np.array(self.real_moves)[to_boxes_map[action[2]]].tolist(),
                            check_numa_action_list=np.array(self.real_numa_moves)[to_boxes_map[action[2]]].tolist(),
                        )
                        move_flag[i] = not undo_flag
                        new_undo_num += int(undo_flag)
                if not undo_flag:
                    to_boxes_map[action[1]].append(i)
            state_change = new_undo_num > 0
        
        if not_check_list is not None:
            move_flag[not_check_list] = True
        undo_moves = np.array(self.real_moves)[~move_flag].tolist()
        undo_numa_moves = np.array(self.real_numa_moves)[~move_flag]
        self._update_undo_move_infos(move_flag, num_undo_move)
        return move_flag, undo_moves, undo_numa_moves


if __name__ == '__main__':
    algorithm = 'aco'
    parser = ArgumentParser()
    parser.add_argument("--data_root_path", type=str, default='../../data/ecs_data')
    parser.add_argument("--data_name", type=str, default='0')
    parser.add_argument("--data_suffix", type=str, default=algorithm)
    parser.add_argument("--sol_save_root_path", type=str, default=f'../../results/ecs_metaheuristic_{algorithm}/placement_ecs_data')
    parser.add_argument("--logger_root_path", type=str, default=f'../../results/ecs_metaheuristic_{algorithm}/moveseq_ecs_data')
    
    parser.add_argument("--is_solve_cycle", action='store_false', default=True)
    parser.add_argument("--is_undo_neg", action='store_false', default=True)
    parser.add_argument("--is_sorted_moves", action='store_false', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.logger_root_path, exist_ok=True)
    
    def run_moveseq(args, data):
        args.data_name = data
        moveseq = Placement2Moveseq(args)
        moveseq.placement_to_moveseq()
    def error_callback(e):
        print(e)
    
    data_list = os.listdir(args.data_root_path)

    pool = Pool(processes=16)
    for data in data_list:
        pool.apply_async(run_moveseq, args=(args, data, ), error_callback=error_callback)
    pool.close()
    pool.join()

