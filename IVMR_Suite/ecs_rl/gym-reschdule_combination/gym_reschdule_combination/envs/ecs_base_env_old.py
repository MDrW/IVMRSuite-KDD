import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import random
import os
import json
import time
from scipy import sparse
import pdb
import gym
from gym import spaces

resource_types = ['memory', 'cpu', 'CPU_BASELINE', 'VM_COUNT', 'VPORT']
max_item_move_count = 1
invalid_action_punish = -10.0
bad_done_punish = -100.0
invalid_action_done_count = 1


class ECS_v1(gym.Env):
    def __init__(self, data_path, is_limited_count=True):
        data_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy/{data_path}"
        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)
        self.is_limited_count = is_limited_count
        
        self.num_items = len(self.item_infos)
        self.num_boxes = len(self.box_infos)


        # 计算均值、标准差、最大值和最小值并存储
        self.pm_cpu_mean = self.box_infos['cpu'].mean()
        self.pm_cpu_std = self.box_infos['cpu'].std()
        self.pm_cpu_max = self.box_infos['cpu'].max()
        self.pm_cpu_min = self.box_infos['cpu'].min()

        self.pm_mem_mean = self.box_infos['memory'].mean()
        self.pm_mem_std = self.box_infos['memory'].std()
        self.pm_mem_max = self.box_infos['memory'].max()
        self.pm_mem_min = self.box_infos['memory'].min()

        self.pm_cpu_baseline_mean = self.box_infos['CPU_BASELINE'].mean()
        self.pm_cpu_baseline_std = self.box_infos['CPU_BASELINE'].std()
        self.pm_cpu_baseline_max = self.box_infos['CPU_BASELINE'].max()
        self.pm_cpu_baseline_min = self.box_infos['CPU_BASELINE'].min()

        self.pm_vm_count_mean = self.box_infos['VM_COUNT'].mean()
        self.pm_vm_count_std = self.box_infos['VM_COUNT'].std()
        self.pm_vm_count_max = self.box_infos['VM_COUNT'].max()
        self.pm_vm_count_min = self.box_infos['VM_COUNT'].min()

        self.pm_vport_mean = self.box_infos['VPORT'].mean()
        self.pm_vport_std = self.box_infos['VPORT'].std()
        self.pm_vport_max = self.box_infos['VPORT'].max()
        self.pm_vport_min = self.box_infos['VPORT'].min()

        self.vm_cpu_mean = self.item_infos['cpu'].mean()
        self.vm_cpu_std = self.item_infos['cpu'].std()
        self.vm_cpu_max = self.item_infos['cpu'].max()
        self.vm_cpu_min = self.item_infos['cpu'].min()

        self.vm_mem_mean = self.item_infos['memory'].mean()
        self.vm_mem_std = self.item_infos['memory'].std()
        self.vm_mem_max = self.item_infos['memory'].max()
        self.vm_mem_min = self.item_infos['memory'].min()

        self.vm_cpu_baseline_mean = self.item_infos['CPU_BASELINE'].mean()
        self.vm_cpu_baseline_std = self.item_infos['CPU_BASELINE'].std()
        self.vm_cpu_baseline_max = self.item_infos['CPU_BASELINE'].max()
        self.vm_cpu_baseline_min = self.item_infos['CPU_BASELINE'].min()

        self.vm_count_mean = self.item_infos['VM_COUNT'].mean()
        self.vm_count_std = self.item_infos['VM_COUNT'].std()
        self.vm_count_max = self.item_infos['VM_COUNT'].max()
        self.vm_count_min = self.item_infos['VM_COUNT'].min()

        self.vm_vport_mean = self.item_infos['VPORT'].mean()
        self.vm_vport_std = self.item_infos['VPORT'].std()
        self.vm_vport_max = self.item_infos['VPORT'].max()
        self.vm_vport_min = self.item_infos['VPORT'].min()

        if self.vm_count_std == 0:
            self.vm_count_std = 1 
        self.vm_vport_mean = self.item_infos['VPORT'].mean()
        self.vm_vport_std = self.item_infos['VPORT'].std()
        
        self.observation_space = spaces.Dict({
            "vm_info": spaces.Box(low=0, high=1, shape=(self.num_items, 9)), 
            "pm_info": spaces.Box(low=0, high=1, shape=(self.num_boxes, 7)),
            "num_steps": spaces.Box(low=0, high=1, shape=(1, 1)),
            "num_vms": spaces.Discrete(self.num_items),
            # "edges": spaces.Box(low=0, high=self.num_vms + self.num_boxes, shape=(self.num_items, 2)),  # edges information
        })
        self.action_space = spaces.MultiDiscrete([self.num_items, self.num_boxes])

        self._init_reward_scale()

        self.seed(0)
        self.reset()
    

    def reset(self):
        # item_cur_state: max_move_count, is_infinite, canMigrate
        self.item_cur_state = self.item_infos[['count', 'isInfinite', 'canMigrate', 'migrationCost'] + resource_types].values
        self.item_cur_state[:, 0] = max_item_move_count
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + resource_types].values
        self.cur_placement = np.array(self.init_placement).copy()
        self.box_remain_resources = self.box_infos[resource_types].values - self.cur_placement.T.dot(self.item_infos[resource_types].values)
        self.box_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        assert (self.box_remain_resources>=0).all(), f"Box Resource Constraints Unsatisfied! Index: {np.where(self.box_remain_resources<0)}"
        self.box_cur_state[:, -len(resource_types):] = self.box_remain_resources
        
        fixed_placement = ((self.item_cur_state[:, 0] <= 0) | (self.item_cur_state[:, 2] <= 0)).astype(float)
        self.box_fixed_remain_resources = self.box_infos[resource_types].values - fixed_placement.T.dot(self.item_infos[resource_types].values)
        self.box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = self.box_infos[resource_types].values.max(axis=0) * 2
        
        self.item_cur_box = np.zeros(len(self.item_cur_state), dtype=int)
        self.infinite_item_cur_box = np.zeros((len(self.item_cur_state), len(self.box_cur_state)), dtype=int)
        self.box_dict = {}
        for i in range(len(self.box_infos)):
            self.box_dict[self.box_infos.loc[i, 'id']] = i
        for i in range(len(self.item_cur_state)):
            self.item_cur_box[i] = self.box_dict[self.item_infos.loc[i, 'inBox']]
        
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        self.actions_available = ((self.cur_placement + self.item_mutex_box == 0) & resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0

        self.invalid_action_count, self.move_count = 0, 0
        self.current_step = 0
        self.bad_done = False

        self.vm_to_pm_cost_matrix = np.zeros((len(self.item_infos), len(self.box_infos)))
        self._calculate_vm_to_pm_cost_matrix()

        # self.edges = np.zeros((self.num_items, 2), dtype=int)
        # for i in range(self.num_items):
        #     self.edges[i, 0] = i  
        #     self.edges[i, 1] = self.item_cur_box[i] 

        # return self._get_state()
        return self._get_dict_state()



    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)


    def step(self, action):
        self.bad_done = False
        action = action.tolist()
        action_available, action_info = self.is_action_available(action)
        self.invalid_action_count += int(not action_available)
        self.move_count += 1

        if action_available:
            costs = self._get_costs(action)
            reward = self._get_reward(costs)
            self._update_state(action)
        else:
            # 处理无效动作
            costs, reward = {'migration_cost': 1e10, 'box_used_cost': 1e10, 'item_assign_cost': 1e10}, invalid_action_punish
            # raise ValueError(f"Invalid Action: {action}, {action_info}")

        state = self._get_dict_state()
        done = self._is_done()
        if self.bad_done:
            reward += bad_done_punish

        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
        }
        infos.update(costs)
        if done:
            done_score = {'done_score': self.get_cur_state_score()}
            infos.update(done_score)
        self.current_step += 1 
        # self.edges[action[0], 0] = action[0]
        # self.edges[action[0], 1] = action[1]

        return state, reward, done, infos

    def _calculate_vm_to_pm_cost_matrix(self):
        for i in range(self.num_items):
            for j in range(self.num_boxes):
                self.vm_to_pm_cost_matrix[i, j] = self.item_assign_box_cost[i, j]

    def undo_step(self, action):
        action_available, action_info = self.is_undo_action_available(action)
        self.move_count -= 1
        if action_available:
            # costs = self._get_undo_costs(action)
            costs = self._get_costs(action)
            reward = self._get_reward(costs)
            self._undo_update_state(action)
            print("reward1", reward)
        else:
            costs, reward = {'migration_cost': 1e10, 'box_used_cost': 1e10, 'item_assign_cost': 1e10}, invalid_action_punish
        
        state = self._get_state()
        done = self._is_done()
        
        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
        }
        infos.update(costs)
        
        return state, reward, done, infos
    
    def get_item_cur_box(self, item_i):
        return self.item_cur_box[item_i]
    
    def _update_box_action_available(self, box_list):
        resource_enough = (np.expand_dims(self.box_cur_state[box_list, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        self.actions_available[:, box_list] = ((self.cur_placement[:, box_list] + self.item_mutex_box[:, box_list] == 0) & resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
    
    def _update_state(self, action):
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        # update placement matrix
        if self.item_cur_state[item_i, 1] <= 0:
            self.cur_placement[item_i, box_j] = 0
            self.cur_placement[item_i, box_k] = 1
            self.item_cur_box[item_i] = box_k
        else:
            self.infinite_item_cur_box[item_i, box_k] += 1
        # update box resource
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
                if self.item_cur_state[item_i, 0] <= 1:
                    self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        elif self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
            self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] -= 1

        self._update_box_action_available([box_j, box_k])
    
    def _undo_update_state(self, action):
        item_i, box_j, box_k = action
        
        # update placement matrix
        if self.item_cur_state[item_i, 1] <= 0:
            self.cur_placement[item_i, box_j] = 0
            self.cur_placement[item_i, box_k] = 1
            self.item_cur_box[item_i] = box_k
        else:
            self.infinite_item_cur_box[item_i, box_j] -= 1
        # update box resource
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
                if self.item_cur_state[item_i, 0] <= 0:
                    self.box_fixed_remain_resources[box_j] += self.item_cur_state[item_i, -len(resource_types):]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
        elif self.box_cur_state[box_j, 1] <= 0:
            self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
            self.box_fixed_remain_resources[box_k] += self.item_cur_state[item_i, -len(resource_types):]
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] += 1

        self._update_box_action_available([box_j, box_k])
    
    def action_space(self):
        item_actions = list(range(len(self.item_cur_state)))
        box_actions = list(range(len(self.box_cur_state)))
        return [item_actions, box_actions]
    
    def state_space(self):
        return [len(x) for x in self._get_state()]
    
    def get_available_actions(self):
        return self.actions_available      
    
    def get_vm_mask(self, id=None):
        available_actions = self.get_available_actions()
        return (available_actions.sum(axis=1) == 0).astype(bool)
    
    def get_pm_mask(self, item_id):
        return (np.array(self.get_available_actions()[item_id]) == 0).astype(bool)

    def is_action_available(self, action):
        if (type(action) != list) or (len(action) != 2):
            return False, f"Action is Invaild, Action Type need be a List of length 2 [item_id, to_box_id]!"
        
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        
        # Box_j equals Box_k?
        if box_j == box_k:
            return False, f"Item {item_i} Departure and Destination are the same box {box_j}!"
        
        # Item_i CanMigrate?
        if self.item_cur_state[item_i, 2] <= 0:
            return False, f"Item {item_i} Can not Migrate!"
        if self.item_cur_state[item_i, 0] <= 0:
            return False, f"Item {item_i} has been Moved {max_item_move_count} Times!"
        
        # Item_i Current In Box_j?
        if self.cur_placement[item_i, box_j] <= 0:
            return False, f"Item {item_i} is not in Box {box_j}!"
        
        # Item_i Mutex Box_k?
        if self.item_mutex_box[item_i, box_k] >= 1:
            return False, f"Item {item_i} Mutex Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(resource_types):] < self.item_cur_state[item_i, -len(resource_types):]).any():
                return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        return True, f"Item {item_i} can move from Box {box_j} to Box {box_k}!"
    
    def is_undo_action_available(self, action):
        if (type(action) != list) or (len(action) != 3):
            return False, f"Action is Invaild, Action Type need be a List of length 3 [item_id, from_box_id, to_box_id]!"
        
        item_i, box_j, box_k = action
        is_item_inf = self.item_cur_state[item_i, 1] >= 1
        cur_box = box_k if is_item_inf else box_j
        
        # Box_j equals Box_k?
        if box_j == box_k:
            return False, f"Item {item_i} Departure and Destination are the same box {box_j}!"
        
        # Item_i CanMigrate?
        if self.item_cur_state[item_i, 2] <= 0:
            return False, f"Item {item_i} Can not Migrate!"
        
        # Item_i Current In Box_j?
        if self.cur_placement[item_i, cur_box] <= 0:
            return False, f"Item {item_i} is not in Box {cur_box}!"
        if is_item_inf and (self.infinite_item_cur_box[item_i, box_j] <= 0):
            return False, f"Box {box_j} does not contain Inifite Item {item_i}!"
        
        # Item_i Mutex Box_k?
        if self.item_mutex_box[item_i, box_k] >= 1:
            return False, f"Item {item_i} Mutex Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(resource_types):] < self.item_cur_state[item_i, -len(resource_types):]).any():
            return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        return True, f"Item {item_i} can move from Box {box_j} to Box {box_k}!"
    
    def _get_costs(self, action):
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        migration_cost = self.item_cur_state[item_i, 3]
        
        item_assign_cost = self.item_assign_box_cost[item_i, box_k] - self.item_assign_box_cost[item_i, box_j]
        
        is_box_k_not_used = int((self.cur_placement[:, box_k].sum() + self.infinite_item_cur_box[:, box_k].sum()) <= 0)
        box_used_cost = self.box_cur_state[box_k, 0] * is_box_k_not_used
        is_box_j_used = int((self.cur_placement[:, box_j].sum() + self.infinite_item_cur_box[:, box_j].sum()) == 1)
        box_used_cost -= self.box_cur_state[box_j, 0] * is_box_j_used
        return {
            'migration_cost': migration_cost,
            'item_assign_cost': item_assign_cost,
            'box_used_cost': box_used_cost,
        }
        
    def _get_undo_costs(self, action):
        item_i, box_j, box_k = action
        
        migration_cost = -self.item_cur_state[item_i, 3]
        
        item_assign_cost = self.item_assign_box_cost[item_i, box_k] - self.item_assign_box_cost[item_i, box_j]
        
        is_box_k_not_used = int((self.cur_placement[:, box_k].sum() + self.infinite_item_cur_box[:, box_k].sum()) <= 0)
        box_used_cost = self.box_cur_state[box_k, 0] * is_box_k_not_used
        is_box_j_used = int((self.cur_placement[:, box_j].sum() + self.infinite_item_cur_box[:, box_j].sum()) == 1)
        box_used_cost -= self.box_cur_state[box_j, 0] * is_box_j_used
        
        return {
            'migration_cost': migration_cost,
            'item_assign_cost': item_assign_cost,
            'box_used_cost': box_used_cost,
        }
    def _init_reward_scale(self):
        def _get_cost_scale(cost, cal_gap=False):
            # cost = np.log(np.abs(cost) + 1)
            if cal_gap:
                max_vals = np.max(cost) - np.min(cost)
            else:
                max_vals = np.max(np.abs(cost))
            return [-max_vals, max_vals]
        migration_scale = self.item_infos['migrationCost'].values
        self.migration_scale = _get_cost_scale(migration_scale)
        
        used_scale = self.box_infos['cost'].values
        self.used_scale = _get_cost_scale(used_scale)
        
        assign_scale = self.item_assign_box_cost
        self.assign_scale = _get_cost_scale(assign_scale, cal_gap=True)

    def _get_reward(self, costs):
        reward = 0.
        for key, value in costs.items():
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
        return reward
    
    def _get_reward_batch(self, box_list):
        item_assign_costs = self.item_assign_box_cost[:, box_list] - (self.item_assign_box_cost * self.cur_placement).sum(axis=1).reshape(-1, 1)
        migration_costs = self.item_cur_state[:, 3:4].repeat(len(box_list), axis=1)
        des_box_used_costs = self.box_cur_state[box_list, 0] * (self.cur_placement[:, box_list].sum(axis=0)<=0)
        origin_box_used_costs = (self.box_cur_state[:, 0] * (self.cur_placement.sum(axis=0)==1))[self.item_cur_box]
        box_used_costs = (des_box_used_costs[None].repeat(len(self.item_cur_state), axis=0) - origin_box_used_costs[:, None].repeat(len(box_list), axis=1))
        reward =  (- item_assign_costs - migration_costs - box_used_costs) # * self.actions_available[:, box_list]
        return reward
    
    def _get_state(self):
        state = [self.item_cur_state, self.box_cur_state, self.cur_placement, self.item_assign_box_cost, self.item_mutex_box]
        return state

    def _get_dict_state(self, mode='mean-std'):
        
        vm_avg_migration_cost = np.mean(self.vm_to_pm_cost_matrix, axis=1)

        if mode == 'mean-std':
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_mean) / self.pm_cpu_std
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_mean) / self.pm_mem_std
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_mean) / self.pm_cpu_baseline_std
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_mean) / self.pm_vm_count_std
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_mean) / self.pm_vport_std
            
            pm_info_norm = np.column_stack([self.box_cur_state[:, :2],
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
                                            pm_vm_count_norm, pm_vport_norm])
            
            vm_cpu_norm = (self.item_cur_state[:, 4] - self.vm_cpu_mean) / self.vm_cpu_std
            vm_mem_norm = (self.item_cur_state[:, 5] - self.vm_mem_mean) / self.vm_mem_std
            vm_cpu_baseline_norm = (self.item_cur_state[:, 6] - self.vm_cpu_baseline_mean) / self.vm_cpu_baseline_std
            vm_vm_count_norm = (self.item_cur_state[:, 7] - self.vm_count_mean) / self.vm_count_std
            vm_vport_norm = (self.item_cur_state[:, 8] - self.vm_vport_mean) / self.vm_vport_std
            
            vm_info_norm = np.column_stack([self.item_cur_state[:, :4],
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
                                            vm_vm_count_norm, vm_vport_norm])
        elif mode == 'max-min':
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_min) / (self.pm_cpu_max - self.pm_cpu_min) if self.pm_cpu_max != self.pm_cpu_min else np.zeros(len(self.box_cur_state))
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_min) / (self.pm_mem_max - self.pm_mem_min) if self.pm_mem_max != self.pm_mem_min else np.zeros(len(self.box_cur_state))
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_min) / (self.pm_cpu_baseline_max - self.pm_cpu_baseline_min) if self.pm_cpu_baseline_max != self.pm_cpu_baseline_min else np.zeros(len(self.box_cur_state))
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_min) / (self.pm_vm_count_max - self.pm_vm_count_min) if self.pm_vm_count_max != self.pm_vm_count_min else np.zeros(len(self.box_cur_state))
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_min) / (self.pm_vport_max - self.pm_vport_min) if self.pm_vport_max != self.pm_vport_min else np.zeros(len(self.box_cur_state))

            pm_info_norm = np.column_stack([self.box_cur_state[:, :2],
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
                                            pm_vm_count_norm, pm_vport_norm])


            vm_cpu_norm = (self.item_cur_state[:, 4] - self.vm_cpu_min) / (self.vm_cpu_max - self.vm_cpu_min) if self.vm_cpu_max != self.vm_cpu_min else np.zeros(len(self.item_cur_state))
            vm_mem_norm = (self.item_cur_state[:, 5] - self.vm_mem_min) / (self.vm_mem_max - self.vm_mem_min) if self.vm_mem_max != self.vm_mem_min else np.zeros(len(self.item_cur_state))
            vm_cpu_baseline_norm = (self.item_cur_state[:, 6] - self.vm_cpu_baseline_min) / (self.vm_cpu_baseline_max - self.vm_cpu_baseline_min) if self.vm_cpu_baseline_max != self.vm_cpu_baseline_min else np.zeros(len(self.item_cur_state))
            vm_vm_count_norm = (self.item_cur_state[:, 7] - self.vm_count_min) / (self.vm_count_max - self.vm_count_min) if self.vm_count_max != self.vm_count_min else np.zeros(len(self.item_cur_state))
            vm_vport_norm = (self.item_cur_state[:, 8] - self.vm_vport_min) / (self.vm_vport_max - self.vm_vport_min) if self.vm_vport_max != self.vm_vport_min else np.zeros(len(self.item_cur_state))

            
            vm_info_norm = np.column_stack([self.item_cur_state[:, :4], vm_avg_migration_cost, 
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
                                            vm_vm_count_norm, vm_vport_norm])


            expand_vm_info_norm = np.zeros((self.max_num_items, vm_info_norm.shape[1]))
            expand_pm_info_norm = np.zeros((self.max_num_boxes, pm_info_norm.shape[1]))
            expand_vm_info_norm[:vm_info_norm.shape[0], :] = vm_info_norm
            expand_pm_info_norm[:pm_info_norm.shape[0], :] = pm_info_norm

        state = {'vm_info': expand_vm_info_norm,
                'pm_info': expand_pm_info_norm,
                'num_steps': self.current_step,
                'num_vms': len(self.item_cur_state)}
        return state


    # def _get_dict_state(self):
    #     # 对pm_info进行归一化
    #     pm_cpu_norm = self.box_cur_state[:, resource_types.index('cpu')] / self.pm_cpu_max
    #     pm_mem_norm = self.box_cur_state[:, resource_types.index('memory')] / self.pm_mem_max
    #     pm_info_norm = np.column_stack([self.box_cur_state[:, :resource_types.index('cpu')],
    #                                     pm_cpu_norm, pm_mem_norm,
    #                                     self.box_cur_state[:, (resource_types.index('cpu')+1):(resource_types.index('memory'))],
    #                                     self.box_cur_state[:, (resource_types.index('memory')+1):]])

    #     # 对vm_info进行归一化
    #     vm_cpu_norm = self.item_cur_state[:, resource_types.index('cpu')] / self.vm_cpu_max
    #     vm_mem_norm = self.item_cur_state[:, resource_types.index('memory')] / self.vm_mem_max
    #     vm_info_norm = np.column_stack([self.item_cur_state[:, :resource_types.index('cpu')],
    #                                     vm_cpu_norm, vm_mem_norm,
    #                                     self.item_cur_state[:, (resource_types.index('cpu')+1):(resource_types.index('memory'))],
    #                                     self.item_cur_state[:, (resource_types.index('memory')+1):]])

    #     state = {'vm_info': self.item_cur_state,
    #              'pm_info': self.box_cur_state,
    #              'num_steps': self.current_step,
    #              'num_vms': len(self.item_cur_state),
    #             }
    #     return state

    def _is_done(self):
        done = False
        if self.invalid_action_count >= invalid_action_done_count:
            done = True
        if self.is_limited_count and (self.move_count >= self.configs['maxMigration']):
            done = True
        if self.actions_available.sum() <= 0:
            done = True
            print("no avaliable action!")
            self.bad_done = True
        
        return done
    
    def render(self, mode='human'):
        pass
    
    def eval_move_sequence(self, action_list):
        self.reset()
        total_reward = 0
        costs = {'migration_cost': 0, 'item_assign_cost': 0, 'box_used_cost': 0}
        invalid_num = 0
        for action in action_list:
            available, info = self.is_action_available(action)
            if available:
                _, reward, done, infos = self.step(action)
                total_reward += reward
                for k in costs.keys():
                    if k in infos.keys():
                        costs[k] += infos[k]
                # print(f"Step {infos['move_count']}: real reward {reward}, done {done}, {infos['action_info']}")
                if done:
                    break
            else:
                invalid_num += 1
                # print(f"NumInv: {invalid_num}, Invalid Action: {action}, {info}")
        # print(f"Total Reward: {total_reward}, InvNum: {invalid_num}")
        return total_reward, costs, invalid_num
    
    def get_cur_state_score(self):
        used_costs = self.box_cur_state[self.cur_placement.sum(axis=0)>0, 0].sum()
        assign_costs = (self.item_assign_box_cost * self.cur_placement).sum()
        return used_costs + assign_costs


class ECS_v2(gym.Env):
    def __init__(self, data_list, mode='train', is_limited_count=True):
        
        self.data_list = data_list
        self.max_num_items = 0
        self.max_num_boxes = 0
        self.mode = mode
        self.num_train_data = min(len(self.data_list) // 5 * 3, 8)
        self.num_eval_data = 1
        for dataset in data_list:
            data_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy/{dataset}"
            item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
            box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
            self.max_num_items = max(self.max_num_items, len(item_infos))
            self.max_num_boxes = max(self.max_num_boxes, len(box_infos))

        
        self.is_limited_count = is_limited_count
        
        self.num_items = self.max_num_items
        self.num_boxes = self.max_num_boxes

        print("num_items", self.num_items) # 2204
        print("num_boxes", self.num_boxes) # 177

        self.observation_space = spaces.Dict({
            "vm_info": spaces.Box(low=0, high=1, shape=(self.num_items, 12)), 
            "pm_info": spaces.Box(low=0, high=1, shape=(self.num_boxes, 10)),
            "num_steps": spaces.Box(low=0, high=1, shape=(1, 1)),
            "num_vms": spaces.Discrete(self.num_items),
            "edges": spaces.Box(0, self.num_items + self.num_boxes, shape=(self.num_items + self.num_boxes, 1)),
        })
        self.action_space = spaces.MultiDiscrete([self.num_items, self.num_boxes])
        self.index = 0
        self.reset()
    
    def set_current_env(self, env_id=None):
        self.index = env_id
    
    def _calculate_vm_to_pm_cost_matrix(self):
        for i in range(len(self.item_infos)):
            for j in range(len(self.box_infos)):
                self.vm_to_pm_cost_matrix[i, j] = self.item_assign_box_cost[i, j]

    def reset(self):
        # selected_dataset = random.choice(self.data_list)
        if self.mode == 'train':
            selected_dataset = self.data_list[self.index]
            self.index = (self.index + 1) % (self.num_train_data)
            print(f"[TRAIN][RESET]: choose dataset:{selected_dataset}")
        elif self.mode == 'eval':
            selected_dataset = self.data_list[self.num_train_data + self.index]
            self.index = (self.index + 1) % (self.num_eval_data)
            print(f"[EVAL][RESET]: choose dataset:{selected_dataset}")
        elif self.mode == 'test':
            selected_dataset = self.data_list[self.index]
            print(f"[TEST][RESET]: choose dataset:{selected_dataset}")

        data_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy/{selected_dataset}"

        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)

        def safe_std(std_value):
            return std_value if std_value != 0 else 1


        self.pm_cpu_mean = self.box_infos['cpu'].mean()
        self.pm_cpu_std = safe_std(self.box_infos['cpu'].std())
        self.pm_cpu_max = self.box_infos['cpu'].max()
        self.pm_cpu_min = self.box_infos['cpu'].min()

        self.pm_mem_mean = self.box_infos['memory'].mean()
        self.pm_mem_std = safe_std(self.box_infos['memory'].std())
        self.pm_mem_max = self.box_infos['memory'].max()
        self.pm_mem_min = self.box_infos['memory'].min()

        self.pm_cpu_baseline_mean = self.box_infos['CPU_BASELINE'].mean()
        self.pm_cpu_baseline_std = safe_std(self.box_infos['CPU_BASELINE'].std())
        self.pm_cpu_baseline_max = self.box_infos['CPU_BASELINE'].max()
        self.pm_cpu_baseline_min = self.box_infos['CPU_BASELINE'].min()

        self.pm_vm_count_mean = self.box_infos['VM_COUNT'].mean()
        self.pm_vm_count_std = safe_std(self.box_infos['VM_COUNT'].std())
        self.pm_vm_count_max = self.box_infos['VM_COUNT'].max()
        self.pm_vm_count_min = self.box_infos['VM_COUNT'].min()

        self.pm_vport_mean = self.box_infos['VPORT'].mean()
        self.pm_vport_std = safe_std(self.box_infos['VPORT'].std())
        self.pm_vport_max = self.box_infos['VPORT'].max()
        self.pm_vport_min = self.box_infos['VPORT'].min()

        self.vm_cpu_mean = self.item_infos['cpu'].mean()
        self.vm_cpu_std = safe_std(self.item_infos['cpu'].std())
        self.vm_cpu_max = self.item_infos['cpu'].max()
        self.vm_cpu_min = self.item_infos['cpu'].min()

        self.vm_mem_mean = self.item_infos['memory'].mean()
        self.vm_mem_std = safe_std(self.item_infos['memory'].std())
        self.vm_mem_max = self.item_infos['memory'].max()
        self.vm_mem_min = self.item_infos['memory'].min()

        self.vm_cpu_baseline_mean = self.item_infos['CPU_BASELINE'].mean()
        self.vm_cpu_baseline_std = safe_std(self.item_infos['CPU_BASELINE'].std())
        self.vm_cpu_baseline_max = self.item_infos['CPU_BASELINE'].max()
        self.vm_cpu_baseline_min = self.item_infos['CPU_BASELINE'].min()

        self.vm_count_mean = self.item_infos['VM_COUNT'].mean()
        self.vm_count_std = safe_std(self.item_infos['VM_COUNT'].std())
        self.vm_count_max = self.item_infos['VM_COUNT'].max()
        self.vm_count_min = self.item_infos['VM_COUNT'].min()

        self.vm_vport_mean = self.item_infos['VPORT'].mean()
        self.vm_vport_std = safe_std(self.item_infos['VPORT'].std())
        self.vm_vport_max = self.item_infos['VPORT'].max()
        self.vm_vport_min = self.item_infos['VPORT'].min()

        self._init_reward_scale()
  
        self.item_cur_state = self.item_infos[['count', 'isInfinite', 'canMigrate', 'migrationCost'] + resource_types].values
        self.item_cur_state[:, 0] = max_item_move_count
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + resource_types].values
        self.cur_placement = np.array(self.init_placement).copy()
        self.box_remain_resources = self.box_infos[resource_types].values - self.cur_placement.T.dot(self.item_infos[resource_types].values)
        self.box_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        assert (self.box_remain_resources>=0).all(), f"Box Resource Constraints Unsatisfied! Index: {np.where(self.box_remain_resources<0)}"
        self.box_cur_state[:, -len(resource_types):] = self.box_remain_resources
        
        fixed_placement = ((self.item_cur_state[:, 0] <= 0) | (self.item_cur_state[:, 2] <= 0)).astype(float)
        self.box_fixed_remain_resources = self.box_infos[resource_types].values - fixed_placement.T.dot(self.item_infos[resource_types].values)
        self.box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = self.box_infos[resource_types].values.max(axis=0) * 2
        
        self.item_cur_box = np.zeros(len(self.item_cur_state), dtype=int)
        self.infinite_item_cur_box = np.zeros((len(self.item_cur_state), len(self.box_cur_state)), dtype=int)
        self.box_dict = {}
        for i in range(len(self.box_infos)):
            self.box_dict[self.box_infos.loc[i, 'id']] = i
        for i in range(len(self.item_cur_state)):
            self.item_cur_box[i] = self.box_dict[self.item_infos.loc[i, 'inBox']]
        
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        self.actions_available = ((self.cur_placement + self.item_mutex_box == 0) & resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0

        self.invalid_action_count, self.move_count = 0, 0
        self.current_step = 0
        self.bad_done = False
        self.done_rewards = 0

        self.vm_to_pm_cost_matrix = np.zeros((len(self.item_infos), len(self.box_infos)))
        self._calculate_vm_to_pm_cost_matrix()

        self.edges = np.expand_dims(np.arange(self.num_items + self.num_boxes), axis=-1)
        for item_id in range(len(self.item_infos)):
            box_id = self.item_cur_box[item_id]
            self.edges[item_id + self.num_boxes, 0] = box_id

        return self._get_dict_state()

    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)


    def step(self, action):
        self.bad_done = False
        action = action.tolist()
        action_available, action_info = self.is_action_available(action)
        self.invalid_action_count += int(not action_available)
        self.move_count += 1

        if action_available:
            costs = self._get_costs(action)
            reward = self._get_reward(costs)
            self._update_state(action)
        else:
            costs, reward = {'migration_cost': 1e10, 'box_used_cost': 1e10, 'item_assign_cost': 1e10}, invalid_action_punish

        state = self._get_dict_state()
        done = self._is_done()
        # if self.bad_done:
        #     reward += bad_done_punish

        self.done_rewards -= costs['migration_cost'] + costs['item_assign_cost'] + costs['box_used_cost']

        self.edges[action[0] + self.num_boxes] = action[1]

        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
        }
        infos.update(costs)
        if done:
            done_score = {'done_score': self.get_cur_state_score()}
            infos.update(done_score)
            done_rewards = {'done_rewards': self.done_rewards}
            infos.update(done_rewards)
        self.current_step += 1 

        return state, reward, done, infos
    
    def get_item_cur_box(self, item_i):
        return self.item_cur_box[item_i]
    
    def _update_box_action_available(self, box_list):
        resource_enough = (np.expand_dims(self.box_cur_state[box_list, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        self.actions_available[:, box_list] = ((self.cur_placement[:, box_list] + self.item_mutex_box[:, box_list] == 0) & resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
    
    def _update_state(self, action):
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        # update placement matrix
        if self.item_cur_state[item_i, 1] <= 0:
            self.cur_placement[item_i, box_j] = 0
            self.cur_placement[item_i, box_k] = 1
            self.item_cur_box[item_i] = box_k
        else:
            self.infinite_item_cur_box[item_i, box_k] += 1
        # update box resource
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
                if self.item_cur_state[item_i, 0] <= 1:
                    self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        elif self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
            self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] -= 1

        self._update_box_action_available([box_j, box_k])
    
    
    def action_space(self):
        item_actions = list(range(len(self.item_cur_state)))
        box_actions = list(range(len(self.box_cur_state)))
        return [item_actions, box_actions]
    
    def state_space(self):
        return [len(x) for x in self._get_state()]
    
    def get_available_actions(self):
        return self.actions_available      
    
    def get_vm_mask(self, id=None):
        available_actions = self.get_available_actions()
        vm_mask = (available_actions.sum(axis=1) == 0).astype(bool)
        expand_vm_mask = np.ones((self.max_num_items)).astype(bool)
        expand_vm_mask[:vm_mask.shape[0]] = vm_mask
        return expand_vm_mask
    
    def get_pm_mask(self, item_id):
        pm_mask = (np.array(self.get_available_actions()[item_id]) == 0).astype(bool)
        expand_pm_mask = np.ones((self.max_num_boxes)).astype(bool)
        expand_pm_mask[:pm_mask.shape[0]] = pm_mask
        return expand_pm_mask


    def is_action_available(self, action):
        if (type(action) != list) or (len(action) != 2):
            return False, f"Action is Invaild, Action Type need be a List of length 2 [item_id, to_box_id]!"
        
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        
        # Box_j equals Box_k?
        if box_j == box_k:
            return False, f"Item {item_i} Departure and Destination are the same box {box_j}!"
        
        # Item_i CanMigrate?
        if self.item_cur_state[item_i, 2] <= 0:
            return False, f"Item {item_i} Can not Migrate!"
        if self.item_cur_state[item_i, 0] <= 0:
            return False, f"Item {item_i} has been Moved {max_item_move_count} Times!"
        
        # Item_i Current In Box_j?
        if self.cur_placement[item_i, box_j] <= 0:
            return False, f"Item {item_i} is not in Box {box_j}!"
        
        # Item_i Mutex Box_k?
        if self.item_mutex_box[item_i, box_k] >= 1:
            return False, f"Item {item_i} Mutex Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(resource_types):] < self.item_cur_state[item_i, -len(resource_types):]).any():
                return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        return True, f"Item {item_i} can move from Box {box_j} to Box {box_k}!"
    
    def is_undo_action_available(self, action):
        if (type(action) != list) or (len(action) != 3):
            return False, f"Action is Invaild, Action Type need be a List of length 3 [item_id, from_box_id, to_box_id]!"
        
        item_i, box_j, box_k = action
        is_item_inf = self.item_cur_state[item_i, 1] >= 1
        cur_box = box_k if is_item_inf else box_j
        
        # Box_j equals Box_k?
        if box_j == box_k:
            return False, f"Item {item_i} Departure and Destination are the same box {box_j}!"
        
        # Item_i CanMigrate?
        if self.item_cur_state[item_i, 2] <= 0:
            return False, f"Item {item_i} Can not Migrate!"
        
        # Item_i Current In Box_j?
        if self.cur_placement[item_i, cur_box] <= 0:
            return False, f"Item {item_i} is not in Box {cur_box}!"
        if is_item_inf and (self.infinite_item_cur_box[item_i, box_j] <= 0):
            return False, f"Box {box_j} does not contain Inifite Item {item_i}!"
        
        # Item_i Mutex Box_k?
        if self.item_mutex_box[item_i, box_k] >= 1:
            return False, f"Item {item_i} Mutex Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(resource_types):] < self.item_cur_state[item_i, -len(resource_types):]).any():
            return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        return True, f"Item {item_i} can move from Box {box_j} to Box {box_k}!"
    
    def _get_costs(self, action):
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        migration_cost = self.item_cur_state[item_i, 3]
        
        item_assign_cost = self.item_assign_box_cost[item_i, box_k] - self.item_assign_box_cost[item_i, box_j]
        
        is_box_k_not_used = int((self.cur_placement[:, box_k].sum() + self.infinite_item_cur_box[:, box_k].sum()) <= 0)
        box_used_cost = self.box_cur_state[box_k, 0] * is_box_k_not_used
        is_box_j_used = int((self.cur_placement[:, box_j].sum() + self.infinite_item_cur_box[:, box_j].sum()) == 1)
        box_used_cost -= self.box_cur_state[box_j, 0] * is_box_j_used
        return {
            'migration_cost': migration_cost,
            'item_assign_cost': item_assign_cost,
            'box_used_cost': box_used_cost,
        }
        
    def _init_reward_scale(self):
        def _get_cost_scale(cost, cal_gap=False):
            # cost = np.log(np.abs(cost) + 1)
            if cal_gap:
                max_vals = np.max(cost) - np.min(cost)
            else:
                max_vals = np.max(np.abs(cost))
            return [-max_vals, max_vals]
        migration_scale = self.item_infos['migrationCost'].values
        self.migration_scale = _get_cost_scale(migration_scale)
        
        used_scale = self.box_infos['cost'].values
        self.used_scale = _get_cost_scale(used_scale)
        
        assign_scale = self.item_assign_box_cost
        self.assign_scale = _get_cost_scale(assign_scale, cal_gap=True)

    def _get_reward(self, costs):
        reward = 0.
        for key, value in costs.items():
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
        return reward
    
    def _get_reward_batch(self, box_list):
        item_assign_costs = self.item_assign_box_cost[:, box_list] - (self.item_assign_box_cost * self.cur_placement).sum(axis=1).reshape(-1, 1)
        migration_costs = self.item_cur_state[:, 3:4].repeat(len(box_list), axis=1)
        des_box_used_costs = self.box_cur_state[box_list, 0] * (self.cur_placement[:, box_list].sum(axis=0)<=0)
        origin_box_used_costs = (self.box_cur_state[:, 0] * (self.cur_placement.sum(axis=0)==1))[self.item_cur_box]
        box_used_costs = (des_box_used_costs[None].repeat(len(self.item_cur_state), axis=0) - origin_box_used_costs[:, None].repeat(len(box_list), axis=1))
        reward =  (- item_assign_costs - migration_costs - box_used_costs) # * self.actions_available[:, box_list]
        return reward
    
    def _get_state(self):
        state = [self.item_cur_state, self.box_cur_state, self.cur_placement, self.item_assign_box_cost, self.item_mutex_box]
        return state

    def _get_dict_state(self, mode='mean-std'):

        vm_mean_migration_cost = np.mean(self.vm_to_pm_cost_matrix, axis=1)
        vm_max_migration_cost = np.max(self.vm_to_pm_cost_matrix, axis=1)
        vm_min_migration_cost = np.min(self.vm_to_pm_cost_matrix, axis=1)

        pm_mean_migration_cost = np.mean(self.vm_to_pm_cost_matrix, axis=0)
        pm_max_migration_cost = np.max(self.vm_to_pm_cost_matrix, axis=0)
        pm_min_migration_cost = np.min(self.vm_to_pm_cost_matrix, axis=0)

        if mode == 'mean-std':
            pm_cpu_std = self.pm_cpu_std if self.pm_cpu_std != 0 else 1  # 避免除以零
            pm_mem_std = self.pm_mem_std if self.pm_mem_std != 0 else 1
            pm_cpu_baseline_std = self.pm_cpu_baseline_std if self.pm_cpu_baseline_std != 0 else 1
            pm_vm_count_std = self.pm_vm_count_std if self.pm_vm_count_std != 0 else 1
            pm_vport_std = self.pm_vport_std if self.pm_vport_std != 0 else 1
            
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_mean) / pm_cpu_std
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_mean) / pm_mem_std
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_mean) / pm_cpu_baseline_std
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_mean) / pm_vm_count_std
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_mean) / pm_vport_std
            
            pm_info_norm = np.column_stack([self.box_cur_state[:, :2], pm_mean_migration_cost, pm_max_migration_cost, pm_min_migration_cost,
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
                                            pm_vm_count_norm, pm_vport_norm])

            vm_cpu_std = self.vm_cpu_std if self.vm_cpu_std != 0 else 1
            vm_mem_std = self.vm_mem_std if self.vm_mem_std != 0 else 1
            vm_cpu_baseline_std = self.vm_cpu_baseline_std if self.vm_cpu_baseline_std != 0 else 1
            vm_vm_count_std = self.vm_count_std if self.vm_count_std != 0 else 1
            vm_vport_std = self.vm_vport_std if self.vm_vport_std != 0 else 1
            
            vm_cpu_norm = (self.item_cur_state[:, 4] - self.vm_cpu_mean) / vm_cpu_std
            vm_mem_norm = (self.item_cur_state[:, 5] - self.vm_mem_mean) / vm_mem_std
            vm_cpu_baseline_norm = (self.item_cur_state[:, 6] - self.vm_cpu_baseline_mean) / vm_cpu_baseline_std
            vm_vm_count_norm = (self.item_cur_state[:, 7] - self.vm_count_mean) / vm_vm_count_std
            vm_vport_norm = (self.item_cur_state[:, 8] - self.vm_vport_mean) / vm_vport_std
            
            vm_info_norm = np.column_stack([self.item_cur_state[:, :4], vm_mean_migration_cost, vm_max_migration_cost, vm_min_migration_cost,
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
                                            vm_vm_count_norm, vm_vport_norm])
        elif mode == 'max-min':
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_min) / (self.pm_cpu_max - self.pm_cpu_min) if self.pm_cpu_max != self.pm_cpu_min else np.zeros(len(self.box_cur_state))
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_min) / (self.pm_mem_max - self.pm_mem_min) if self.pm_mem_max != self.pm_mem_min else np.zeros(len(self.box_cur_state))
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_min) / (self.pm_cpu_baseline_max - self.pm_cpu_baseline_min) if self.pm_cpu_baseline_max != self.pm_cpu_baseline_min else np.zeros(len(self.box_cur_state))
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_min) / (self.pm_vm_count_max - self.pm_vm_count_min) if self.pm_vm_count_max != self.pm_vm_count_min else np.zeros(len(self.box_cur_state))
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_min) / (self.pm_vport_max - self.pm_vport_min) if self.pm_vport_max != self.pm_vport_min else np.zeros(len(self.box_cur_state))

            pm_info_norm = np.column_stack([self.box_cur_state[:, :2], pm_mean_migration_cost, pm_max_migration_cost, pm_min_migration_cost,
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
                                            pm_vm_count_norm, pm_vport_norm])


            vm_cpu_norm = (self.item_cur_state[:, 4] - self.vm_cpu_min) / (self.vm_cpu_max - self.vm_cpu_min) if self.vm_cpu_max != self.vm_cpu_min else np.zeros(len(self.item_cur_state))
            vm_mem_norm = (self.item_cur_state[:, 5] - self.vm_mem_min) / (self.vm_mem_max - self.vm_mem_min) if self.vm_mem_max != self.vm_mem_min else np.zeros(len(self.item_cur_state))
            vm_cpu_baseline_norm = (self.item_cur_state[:, 6] - self.vm_cpu_baseline_min) / (self.vm_cpu_baseline_max - self.vm_cpu_baseline_min) if self.vm_cpu_baseline_max != self.vm_cpu_baseline_min else np.zeros(len(self.item_cur_state))
            vm_vm_count_norm = (self.item_cur_state[:, 7] - self.vm_count_min) / (self.vm_count_max - self.vm_count_min) if self.vm_count_max != self.vm_count_min else np.zeros(len(self.item_cur_state))
            vm_vport_norm = (self.item_cur_state[:, 8] - self.vm_vport_min) / (self.vm_vport_max - self.vm_vport_min) if self.vm_vport_max != self.vm_vport_min else np.zeros(len(self.item_cur_state))

            vm_info_norm = np.column_stack([self.item_cur_state[:, :4], vm_mean_migration_cost, vm_max_migration_cost, vm_min_migration_cost,
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
                                            vm_vm_count_norm, vm_vport_norm])
        
        expand_vm_info_norm = np.zeros((self.max_num_items, vm_info_norm.shape[1]))
        expand_pm_info_norm = np.zeros((self.max_num_boxes, pm_info_norm.shape[1]))
        expand_vm_info_norm[:vm_info_norm.shape[0], :] = vm_info_norm
        expand_pm_info_norm[:pm_info_norm.shape[0], :] = pm_info_norm

        state = {'vm_info': expand_vm_info_norm,
                'pm_info': expand_pm_info_norm,
                'num_steps': self.current_step,
                'num_vms': len(self.item_cur_state),
                'edges': self.edges}
        return state

    def _is_done(self):
        done = False
        if self.invalid_action_count >= invalid_action_done_count:
            done = True
        if self.is_limited_count and (self.move_count >= self.configs['maxMigration']):
            done = True
        if self.actions_available.sum() <= 0:
            done = True
            print("no avaliable action!")
            self.bad_done = True
        
        return done
    
    def render(self, mode='human'):
        pass
    
    def eval_move_sequence(self, action_list):
        self.reset()
        total_reward = 0
        costs = {'migration_cost': 0, 'item_assign_cost': 0, 'box_used_cost': 0}
        invalid_num = 0
        for action in action_list:
            available, info = self.is_action_available(action)
            if available:
                _, reward, done, infos = self.step(action)
                total_reward += reward
                for k in costs.keys():
                    if k in infos.keys():
                        costs[k] += infos[k]
                if done:
                    break
            else:
                invalid_num += 1
        return total_reward, costs, invalid_num
    
    def get_cur_state_score(self):
        used_costs = self.box_cur_state[self.cur_placement.sum(axis=0)>0, 0].sum()
        assign_costs = (self.item_assign_box_cost * self.cur_placement).sum()
        return used_costs + assign_costs


if __name__ == '__main__':
    data_path = '../data/nocons_finite_easy/8621'
    s = time.time()
    env = ECSEnvironment(data_path=data_path)
    print(time.time() - s)
