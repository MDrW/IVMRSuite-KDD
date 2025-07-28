import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym.spaces
from ecs_env.ecs_base_env import ECSEnvironment
import numpy as np
import gym
from scipy import sparse
import pandas as pd
# from multiprocess import Pool
from copy import deepcopy as dcp
import pdb
import time


class ECSRawDenseEnvironment:
    def __init__(self, data_path, is_limited_count=True, is_filter_unmovable=True,
                 is_dense_items=True, is_dense_boxes=False, 
                 is_state_merge_placement=False, is_normalize_state=True, is_process_numa=False):
        self.ecs_env = ECSEnvironment(data_path=data_path, is_limited_count=is_limited_count, 
                                      is_filter_unmovable=is_filter_unmovable, is_process_numa=is_process_numa)
        
        self.is_dense_items = is_dense_items
        self.is_dense_boxes = is_dense_boxes
        self.is_state_merge_placement = is_state_merge_placement
        self.is_normalize_state = is_normalize_state
        
        self._seed = 0
        self.item_count, self.box_count = len(self.ecs_env.item_infos), len(self.ecs_env.box_infos)
        
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        self.pre_actions = []
        
        self.resource_types = self.ecs_env.resource_types
        
    def _init_reward_scale(self):
        def _get_cost_scale(cost, cal_gap=False):
            # cost = np.log(np.abs(cost) + 1)
            if cal_gap:
                max_vals = np.max(cost) - np.min(cost)
            else:
                max_vals = np.max(np.abs(cost))
            return [-max_vals, max_vals]
        migration_scale = self.ecs_env.item_infos['migrationCost'].values
        self.migration_scale = _get_cost_scale(migration_scale)
        
        used_scale = self.ecs_env.box_infos['cost'].values
        self.used_scale = _get_cost_scale(used_scale)
        
        assign_scale = self.ecs_env.item_assign_box_cost
        self.assign_scale = _get_cost_scale(assign_scale, cal_gap=True)
        # print(np.max(assign_scale), np.min(assign_scale), self.assign_scale)
    
    def _init_state_scale(self):
        # self.item_state_max, self.item_state_min = self.ecs_env.item_cur_state.max(axis=0), self.ecs_env.item_cur_state.min(axis=0)
        self.box_state_max, self.box_state_min = self.ecs_env.box_cur_state.max(axis=0), self.ecs_env.box_cur_state.min(axis=0)
        
        items_state = np.concatenate([
            self.ecs_env.item_cur_state,
            self.ecs_env.item_assign_box_cost.max(axis=-1).reshape(-1, 1), 
            self.ecs_env.item_assign_box_cost.min(axis=-1).reshape(-1, 1),
            self.ecs_env.item_assign_box_cost.mean(axis=-1).reshape(-1, 1),
            self.ecs_env.item_cur_box.reshape(-1, 1),
        ], axis=-1)
        boxes_state = np.concatenate([
            self.ecs_env.box_cur_state,
            self.ecs_env.item_assign_box_cost.max(axis=0).reshape(-1, 1), 
            self.ecs_env.item_assign_box_cost.min(axis=0).reshape(-1, 1),
            self.ecs_env.item_assign_box_cost.mean(axis=0).reshape(-1, 1),
        ], axis=-1)
        
        self.dense_items_map, self.dense_items_idxes = self._dense_items(items_state)
        
        self.dense_boxes_map, self.dense_boxes_idxes = self._dense_boxes(boxes_state)
        
        cur_placement, item_assign_cost, item_mutex_box = self.ecs_env.cur_placement, self.ecs_env.item_assign_box_cost, self.ecs_env.item_mutex_box
        available_actions = self.get_available_action()
        item_assign_cost = (item_assign_cost) / (self.assign_scale[1] + 1e-5)
        if self.is_dense_items:
            newi_cur_placement, newi_item_assign_cost, newi_item_mutex_box, newi_available_actions = [0] * len(self.dense_items_map), [0] * len(self.dense_items_map), [0] * len(self.dense_items_map), [0] * len(self.dense_items_map)
            newi_placement_num = [0] * len(self.dense_items_map)
            dense_items_state = [0] * len(self.dense_items_map)
            def _process_dense_items(i, items_idxes):
                newi_cur_placement[i] = cur_placement[items_idxes].max(axis=0).tolist()
                newi_placement_num[i] = cur_placement[items_idxes].sum(axis=0).tolist()
                newi_item_assign_cost[i] = item_assign_cost[items_idxes].max(axis=0).tolist()
                newi_item_mutex_box[i] = item_mutex_box[items_idxes].min(axis=0).tolist()
                newi_available_actions[i] = available_actions[items_idxes].max(axis=0).tolist()
                dense_items_state[i] = items_state[items_idxes].max(axis=0).tolist()
                dense_items_state[i][0] = (items_state[items_idxes, 0] * items_state[items_idxes, 2]).sum()
            for i, items_idxes in self.dense_items_map.items():
                _process_dense_items(i, items_idxes)
        else:
            newi_cur_placement, newi_item_assign_cost, newi_item_mutex_box, newi_available_actions = cur_placement.tolist(), item_assign_cost.tolist(), item_mutex_box.tolist(), available_actions.tolist()
            newi_placement_num, dense_items_state = cur_placement.tolist(), items_state.tolist()
        self.items_count = np.array(dense_items_state)[:, 0]
        self.items_max_count = self.items_count.max()
        items_resources, item_state = np.array(dense_items_state)[:, 4:4+len(self.resource_types)], np.array(dense_items_state)[:, [1, 2, 3, -3, -2, -1]]
        if self.is_normalize_state:
            item_state = (item_state - item_state.min(axis=0)) / (item_state.max(axis=0) - item_state.min(axis=0) + 1e-5)
            items_resources = items_resources / (self.box_state_max[-len(self.resource_types):] + 1e-5)
            self.items_count /= self.items_max_count
        self.dense_items_state = np.concatenate([
            self.items_count.reshape(-1, 1),
            item_state,
            items_resources,
        ], axis=-1)
        
        if self.is_normalize_state:
            box_fixed_remain_resources = self.ecs_env.box_fixed_remain_resources / (self.box_state_max[-len(self.resource_types):] + 1e-5)
            boxes_state = boxes_state / (boxes_state.max(axis=0) + 1e-5)
        else:
            box_fixed_remain_resources = self.ecs_env.box_fixed_remain_resources
        boxes_state = np.concatenate([
            boxes_state,
            box_fixed_remain_resources,
        ], axis=-1)
        
        if self.is_dense_boxes:
            new_cur_placement, new_item_assign_cost, new_item_mutex_box, new_available_actions = [0] * len(self.dense_boxes_map), [0] * len(self.dense_boxes_map), [0] * len(self.dense_boxes_map), [0] * len(self.dense_boxes_map)
            new_placement_num = [0] * len(self.dense_boxes_map)
            dense_boxes_state = [0] * len(self.dense_boxes_map)
            # boxes_state = (boxes_state - boxes_state.min(axis=0)) / (boxes_state.max(axis=0) - boxes_state.min(axis=0) + 1e-5)
            def _process_dense_boxes(i, boxes_idxes):
                new_cur_placement[i] = np.array(newi_cur_placement)[:, boxes_idxes].max(axis=-1).tolist()
                new_placement_num[i] = np.array(newi_placement_num)[:, boxes_idxes].sum(axis=-1).tolist()
                new_item_assign_cost[i] = np.array(newi_item_assign_cost)[:, boxes_idxes].max(axis=-1).tolist()
                new_item_mutex_box[i] = np.array(newi_item_mutex_box)[:, boxes_idxes].min(axis=-1).tolist()
                new_available_actions[i] = np.array(newi_available_actions)[:, boxes_idxes].max(axis=-1).tolist()
                dense_boxes_state[i] = boxes_state[boxes_idxes].sum(axis=0).tolist()
            for i, boxes_idxes in self.dense_boxes_map.items():
                _process_dense_boxes(i, boxes_idxes)
            self.dense_boxes_state = np.array(dense_boxes_state)
            
            self.dense_cur_placement = np.array(new_cur_placement).T
            self.dense_placement_num = np.array(new_placement_num).T
            self.dense_available_actions = np.array(new_available_actions).T
            self.dense_item_assign_cost = sparse.csr_matrix(np.array(new_item_assign_cost).T)
            self.dense_item_mutex_box = sparse.csr_matrix(np.array(new_item_mutex_box).T)
        else:
            self.dense_boxes_state = np.array(boxes_state)
            self.dense_cur_placement = np.array(newi_cur_placement)
            self.dense_placement_num = np.array(newi_placement_num)
            self.dense_available_actions = np.array(newi_available_actions)
            self.dense_item_assign_cost = sparse.csr_matrix(np.array(newi_item_assign_cost))
            self.dense_item_mutex_box = sparse.csr_matrix(np.array(newi_item_mutex_box))
    
    def _dense_items(self, items_state, cols=None, groupby=None):
        if cols is None:
            cols = ['count', 'isInfinite', 'canMigrate', 'migrationCost'] + self.resource_types + ['amax', 'amin', 'amean', 'inbox']
        if groupby is None:
            groupby = ['migrationCost'] + self.resource_types + ['amax', 'amin', 'amean', 'inbox']
        if self.is_dense_items:
            items_pd = pd.DataFrame(items_state, columns=cols)
            
            i, dense_items_idxes, items_map = 0, np.ones(len(items_state), dtype=int) * -1, {}
            for _, g in items_pd.groupby(by=groupby):
                items_map[i] = g.index.tolist()
                dense_items_idxes[g.index.tolist()] = i
                i += 1
        else:
            # items_map = {i: [i] for i in range(len(items_state))}
            dense_items_idxes = np.arange(len(items_state), dtype=int)
            items_map = dict(zip(dense_items_idxes.tolist(), dense_items_idxes.reshape(-1, 1).tolist()))
        return items_map, dense_items_idxes
    
    def _dense_boxes(self, boxes_state, cols=None, groupby=None):
        if cols is None:
            cols = ['cost', 'isInfinite'] + self.resource_types + ['bmax', 'bmin', 'bmean']
        if groupby is None:
            groupby = ['cost', 'isInfinite'] + self.resource_types + ['bmax', 'bmin', 'bmean']
        if self.is_dense_boxes:
            boxes_pd = pd.DataFrame(boxes_state, columns=cols)
            
            i, dense_boxes_idxes, boxes_map = 0, np.ones(len(boxes_state), dtype=int) * -1, {}
            for _, g in boxes_pd.groupby(by=groupby):
                boxes_map[i] = g.index.tolist()
                dense_boxes_idxes[g.index.tolist()] = i
                i += 1
        else:
            # boxes_map = {i: [i] for i in range(len(boxes_state))}
            dense_boxes_idxes = np.arange(len(boxes_state), dtype=int)
            boxes_map = dict(zip(dense_boxes_idxes.tolist(), dense_boxes_idxes.reshape(-1, 1).tolist()))
        return boxes_map, dense_boxes_idxes
    
    def _process_state(self, full_state, action=None):
        item_state, box_state = self.dense_items_state, self.dense_boxes_state
        item_assign_cost, item_mutex_box = self.dense_item_assign_cost, self.dense_item_mutex_box
        cur_placement, available_actions = self.dense_cur_placement, self.dense_available_actions
        if action is not None:
            item_i, box_j, box_k = action
            box_fixed_remain_resources = self.ecs_env.box_fixed_remain_resources / (self.box_state_max[-len(self.resource_types):] + 1e-5)
            boxes_state = np.concatenate([
                full_state[1],
                full_state[3].max(axis=0).reshape(-1, 1), 
                full_state[3].min(axis=0).reshape(-1, 1),
                full_state[3].mean(axis=0).reshape(-1, 1),
            ], axis=1)
            boxes_state = np.concatenate([
                boxes_state / (boxes_state.max(axis=0) + 1e-5),
                box_fixed_remain_resources,
            ], axis=-1)
            j_i, k_i = self.dense_boxes_idxes[box_j], self.dense_boxes_idxes[box_k]
            box_state[j_i] = boxes_state[self.dense_boxes_map[j_i]].sum(axis=0)
            box_state[k_i] = boxes_state[self.dense_boxes_map[k_i]].sum(axis=0)
            
            i_i = self.dense_items_idxes[item_i]
            self.items_count[i_i] -= 1
            item_state[i_i, 0] = self.items_count[i_i]
            if self.is_normalize_state:
                item_state[i_i, 0] /= self.items_max_count
            
            cur_placement[i_i, j_i] = full_state[2][self.dense_items_map[i_i]][:, self.dense_boxes_map[j_i]].max()
            cur_placement[i_i, k_i] = 1
            self.dense_placement_num[i_i, j_i] -= 1
            self.dense_placement_num[i_i, k_i] += 1
            
            cur_available = self.get_available_action()
            # pdb.set_trace()
            available_j = self.dense_items_idxes[(cur_available[:, self.dense_boxes_map[j_i]].max(axis=-1))>=1].astype(int).tolist()
            available_k = self.dense_items_idxes[(cur_available[:, self.dense_boxes_map[k_i]].max(axis=-1))>=1].astype(int).tolist()
            available_i = self.dense_boxes_idxes[(cur_available[self.dense_items_map[i_i], :].max(axis=0))>=1].astype(int).tolist()
            available_actions[:, [j_i, k_i]] = 0
            available_actions[list(set(available_j)), j_i] = 1
            available_actions[list(set(available_k)), k_i] = 1
            available_actions[i_i, :] = 0
            available_actions[i_i, list(set(available_i))] = 1
            
            self.dense_cur_placement, self.dense_available_actions = cur_placement, available_actions
            self.dense_items_state, self.dense_boxes_state = item_state, box_state
            
        if self.is_state_merge_placement:
            items_state_in_same_box = self.dense_placement_num.T.dot(self.dense_items_state) / (self.dense_placement_num.T.sum(axis=-1).reshape(-1, 1) + 1e-5)
            box_state = np.concatenate([
                self.dense_boxes_state,
                items_state_in_same_box,
            ], axis=-1)
            boxes_state_for_item = self.dense_placement_num.dot(self.dense_boxes_state) / (self.dense_placement_num.sum(axis=-1).reshape(-1, 1) + 1e-5)
            item_state = np.concatenate([
                self.dense_items_state,
                boxes_state_for_item,
            ], axis=-1)
            
        return item_state.copy(), box_state.copy(), sparse.csr_matrix(cur_placement.copy()), dcp(item_assign_cost), dcp(item_mutex_box), sparse.csr_matrix(available_actions.copy()), dcp(self.pre_actions)
    
    def get_available_action(self):
        return np.array(self.ecs_env.get_available_actions())
    
    def get_available_item(self):
        available_actions = self.ecs_env.get_available_actions()
        return (available_actions.sum(axis=1) > 0).astype(int)
    
    def get_item_available_action(self, item_i):
        return np.array(self.ecs_env.get_available_actions()[item_i])
    
    def get_available_box(self):
        available_actions = self.ecs_env.get_available_actions()
        return (available_actions.sum(axis=0) > 0).astype(int)
    
    def get_box_available_action(self, box_j):
        return np.array(self.ecs_env.get_available_actions()[:, box_j])
    
    def _process_dense_action(self, action):
        item_idxes = self.dense_items_map[action[0]]
        box_idxes = self.dense_boxes_map[action[1]]
        item_count, box_count = len(item_idxes), len(box_idxes)
        item_idxes = sorted(item_idxes * box_count)
        box_idxes = box_idxes * item_count
        cost = self.ecs_env.item_assign_box_cost.copy()
        cost[self.ecs_env.get_available_actions()==0] = 1e50
        idx = (self.ecs_env.item_assign_box_cost[item_idxes, self.ecs_env.item_cur_box[item_idxes]] - cost[item_idxes, box_idxes]).argmax()
        # pdb.set_trace()
        # if not self.ecs_env.is_action_available([item_idxes[idx], box_idxes[idx]])[0]:
        #     pdb.set_trace()
        return [item_idxes[idx], box_idxes[idx]]
    
    def step(self, action, is_act_dense=False, numa_action=None):
        full_action = [action[0], -1, action[1]]
        if is_act_dense:
            action = self._process_dense_action(action)
        full_action[1] = self.dense_boxes_idxes[self.ecs_env.item_cur_box[action[0]]]
        box_j = self.ecs_env.get_item_cur_box(action[0])
        step_action = [action[0], box_j, action[1]]
        full_state, reward, done, infos = self.ecs_env.step(list(action), numa_action)
        
        reward = self._get_reward(infos)
        infos['real_reward'] = -(infos['migration_cost'] + infos['item_assign_cost'] + infos['box_used_cost'])
        infos['real_action'] = step_action
        state = self._process_state(full_state=full_state, action=step_action)
        self.pre_actions.append(full_action)
        return state, reward, done, infos
    
    def undo_step(self, action):
        state, reward, done, infos = self.ecs_env.undo_step(action)
        reward = self._get_reward(infos)
        infos['real_reward'] = -(infos['migration_cost'] + infos['item_assign_cost'] + infos['box_used_cost'])
        return state, reward, done, infos
    
    def _get_reward(self, reward_dict):
        reward = 0.
        for key, value in reward_dict.items():
            if key == 'migration_cost':
                min_v, max_v = self.migration_scale
                multiplier = 0.1
            elif key == 'box_used_cost':
                min_v, max_v = self.used_scale
                multiplier = 0.5
            elif key == 'item_assign_cost':
                min_v, max_v = self.assign_scale
                multiplier = 10
            else:
                continue
            reward -= multiplier * value / (max_v + 1e-5)
            # reward -= multiplier * np.sign(value) * (np.log(np.abs(value) + 1) - min_v) / (max_v - min_v + 1e-5)
        return reward
    
    def reset(self):
        full_state = self.ecs_env.reset()
        
        self.item_count, self.box_count = len(self.ecs_env.item_infos), len(self.ecs_env.box_infos)
        
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        self._init_reward_scale()
        self._init_state_scale()
        self.pre_actions = []
        state = self._process_state(full_state=full_state)
        return state
    
    def seed(self, _seed):
        self._seed = _seed
        self.ecs_env.seed(_seed)
        self.action_space.seed(_seed)
        self.observation_space.seed(_seed)
    
    def get_action_space(self):
        actions = self.get_available_action()
        action_space = gym.spaces.Discrete(len(actions), seed=self._seed, start=0)
        return action_space
    
    def get_observation_space(self):
        # full_state = self.ecs_env._get_state()
        # state = self._process_state(full_state=full_state)
        state_space = gym.spaces.Box(shape=(self.item_count, self.box_count), seed=self._seed, low=-np.inf, high=np.inf)
        return state_space
    
    def render(self, mode='human'):
        self.ecs_env.render(mode)
    
    def get_cur_state_score(self):
        return self.ecs_env.get_cur_state_score()
    
    def eval_move_sequence(self, action_list, is_act_dense=False, numa_action_list=None):
        self.reset()
        total_reward = 0
        total_real_reward = 0
        invalid_num, num_action = 0, 0
        for ai, action in enumerate(action_list):
            if is_act_dense:
                action = self._process_dense_action(action)
            if numa_action_list is not None:
                numa_action = numa_action_list[ai]
            # available, info = self.ecs_env.is_action_available(action)
            available = self.ecs_env.actions_available[action[0], action[1]] > 0
            if available:
                _, reward, done, infos = self.step(action, False, numa_action)
                total_reward += reward
                total_real_reward += infos['real_reward']
                num_action += 1
                # print(f"Step {infos['move_count']}: real reward {reward}, done {done}, {infos['action_info']}")
                if done:
                    break
            else:
                invalid_num += 1
                # print(f"NumInv: {invalid_num}, Invalid Action: {action}, {info}")
        # print(f"Total Reward: {total_reward}, InvNum: {invalid_num}")
        return total_reward, total_real_reward, num_action, invalid_num
    
    def close(self):
        del self.ecs_env
        del self


if __name__ == '__main__':
    start = time.time()
    data_path = '../data/ecs_data/1673'
    env = ECSRawDenseEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True,
        is_dense_items=True, is_dense_boxes=True, is_state_merge_placement=True)
    env.reset()
    print("Reset Time: ", time.time()-start)
    done = False
    total_reward = 0
    while not done:
        st = time.time()
        available = env.get_available_action()
        # import pdb
        # pdb.set_trace()
        item = np.random.choice(np.where(available.sum(axis=1)>0)[0], size=1)[0]
        box = np.random.choice(np.where(available[item]>0)[0], size=1)[0]
        action = [item, box]
        s, r, done, info = env.step(action)
        total_reward += info['real_reward']
        print(f"Step {info['move_count']}: reward {r}, done {done}, {info['action_info']}, Time: {time.time()-st}s!")
    print(total_reward, time.time()-start)