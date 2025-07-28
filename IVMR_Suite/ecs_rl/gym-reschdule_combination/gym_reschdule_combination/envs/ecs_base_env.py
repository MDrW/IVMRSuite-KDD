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
from copy import deepcopy as dcp

resource_types = ['memory', 'cpu', 'CPU_BASELINE', 'VM_COUNT', 'VPORT']
max_item_move_count = 1
invalid_action_punish = -10.0
bad_done_punish = -100.0
invalid_action_done_count = 1

class ECS_v2(gym.Env):
    def __init__(self, data_list, mode='train', is_limited_count=True, is_filter_unmovable=True, is_process_numa=True):
        
        
        self.data_list = data_list
        self.max_num_items = 0
        self.max_num_boxes = 0
        self.mode = mode
        self.is_process_numa = is_process_numa

        total_datasets = len(self.data_list)
        
        train_indices = []
        eval_indices = []

        for i in range(total_datasets):
            if i % 3 != 2 and len(train_indices) <= 6:
                train_indices.append(i)

            # else:
            #     eval_indices.append(i)
            # train_indices.append(i)
            eval_indices.append(i)

        
        self.train_data = [self.data_list[i] for i in train_indices]
        self.eval_data = [self.data_list[i] for i in eval_indices]

        self.num_train_data = len(train_indices)
        self.num_eval_data = len(eval_indices)

        for dataset in data_list:
            
            # data_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy/{dataset}"
            data_path = f"/mnt/workspace/workgroup/xuwan/datas/novio_finite_noii_v2/{dataset}"
            item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
            box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
            self.max_num_items = max(self.max_num_items, len(item_infos))
            # self.max_num_items = 5000
            self.max_num_boxes = max(self.max_num_boxes, len(box_infos))
            
            with open(os.path.join(data_path, 'config.json'), 'r') as f:
                self.configs = json.load(f)
        
        self.is_limited_count = is_limited_count
        self.is_filter_unmovable = is_filter_unmovable
        

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
        # self.reset()
    
    def set_current_env(self, env_id=None):
        self.index = env_id
    
    def _calculate_vm_to_pm_cost_matrix(self):
        for i in range(self.item_assign_box_cost.shape[0]):
            for j in range(self.item_assign_box_cost.shape[1]):
                self.vm_to_pm_cost_matrix[i, j] = self.item_assign_box_cost[i, j]

    def reset(self):
        if self.mode == 'train':
            selected_dataset = self.train_data[self.index]
            self.index = (self.index + 1) % (self.num_train_data)
            print(f"[TRAIN][RESET]: choose dataset:{selected_dataset}")
        elif self.mode == 'eval':
            selected_dataset = self.eval_data[self.index]
            self.index = (self.index + 1) % (self.num_eval_data)
            print(f"[EVAL][RESET]: choose dataset:{selected_dataset}")
        elif self.mode == 'test':
            selected_dataset = self.data_list[self.index]
            print(f"[TEST][RESET]: choose dataset:{selected_dataset}")

        # data_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy/{selected_dataset}"
        data_path = f"/mnt/workspace/workgroup/xuwan/datas/novio_finite_noii_v2/{selected_dataset}"

        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.item_full_infos = pd.read_pickle(os.path.join(data_path, 'item_infos_v1.pkl'))
        self.box_full_infos = pd.read_pickle(os.path.join(data_path, 'box_infos_v1.pkl'))
        

        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)

        if (self.configs['maxBoxNuma'] > 0) and (self.configs['maxItemNuma'] > 0) and self.is_process_numa:
            self.len_numa_res = len(self.configs['compound_resource_types']['numa'])
            self.max_box_numa = self.configs['maxBoxNuma']
            self.init_boxes_numa = np.zeros((len(self.box_infos), self.max_box_numa*self.len_numa_res), dtype=float)
            self.init_items_numa = np.zeros((len(self.item_infos), self.max_box_numa*self.len_numa_res), dtype=float)
            self.init_items_numa_res = np.zeros((len(self.item_infos), self.len_numa_res), dtype=float)
            self.init_items_numa_num = np.zeros(len(self.item_infos), dtype=int)
            for box_j in range(len(self.box_infos)):
                box_j_numa = self.box_full_infos.iloc[box_j]['numa']
                for ri in range(len(box_j_numa)):
                    self.init_boxes_numa[box_j][self.len_numa_res*ri:self.len_numa_res*(ri+1)] = box_j_numa[ri]
            for item_i in range(len(self.item_infos)):
                item_i_numa = self.item_full_infos.iloc[item_i]['numa']
                for idx in item_i_numa[0]:
                    self.init_items_numa[item_i][int(idx)*self.len_numa_res:(int(idx)+1)*self.len_numa_res] = item_i_numa[2:]
                    self.init_items_numa_num[item_i] = item_i_numa[1]
                    self.init_items_numa_res[item_i] = item_i_numa[2:]
        else:
            self.len_numa_res, self.init_boxes_numa, self.init_items_numa = 0, None, None
        

        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost_origin = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box_origin = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()

        # item_cur_state: max_move_count, is_infinite, canMigrate
        self.item_assign_box_cost, self.item_mutex_box = self.item_assign_box_cost_origin.copy(), self.item_mutex_box_origin.copy()
        self.cur_placement = np.array(self.init_placement).copy()
        self.item_cur_state = self.item_infos[['count', 'isInfinite', 'canMigrate', 'migrationCost'] + resource_types].values
        self.item_cur_state[:, 0] = max_item_move_count
        self.item_init_movable = (self.item_cur_state[:, 0] > 0) & (self.item_cur_state[:, 2] > 0)
        
        self.item_cur_box = np.zeros(len(self.item_cur_state), dtype=int)
        self.infinite_item_cur_box = np.zeros((len(self.item_cur_state), len(self.box_infos)), dtype=int)

        self.box_dict = {}
        for i in range(len(self.box_infos)):
            self.box_dict[self.box_infos.loc[i, 'id']] = i
        for i in range(len(self.item_cur_state)):
            self.item_cur_box[i] = self.box_dict[self.item_infos.loc[i, 'inBox']]
        
        self.item_cur_numa = dcp(self.init_items_numa)
        self.item_new_numa, self.stored_numa_actions = {}, {}
        self.box_cur_numa = dcp(self.init_boxes_numa)
        self.item_numa_res = dcp(self.init_items_numa_res)
        self.item_numa_num = dcp(self.init_items_numa_num)
        

        if self.is_filter_unmovable:
            # print(len(self.item_cur_state), self.item_init_movable.astype(float).sum())
            self.item_cur_state = self.item_cur_state[self.item_init_movable]
            self.item_cur_box = self.item_cur_box[self.item_init_movable]
            self.item_assign_box_cost = self.item_assign_box_cost[self.item_init_movable]
            self.item_mutex_box = self.item_mutex_box[self.item_init_movable]
            self.cur_placement = self.cur_placement[self.item_init_movable]
            
            if self.len_numa_res > 0:
                self.item_cur_numa = self.item_cur_numa[self.item_init_movable]
                self.item_numa_res = self.item_numa_res[self.item_init_movable]
                self.item_numa_num = self.item_numa_num[self.item_init_movable]

            i, self.init_idxes_map, self.filter_to_unfilter = 0, {}, {}
            for ri in range(len(self.item_init_movable)):
                if self.item_init_movable[ri]:
                    self.init_idxes_map[ri] = i
                    self.filter_to_unfilter[i] = ri
                    i += 1
        
        items_state = np.concatenate([
            self.item_cur_state[:, 3:],
            self.item_assign_box_cost.max(axis=-1).reshape(-1, 1), 
            self.item_assign_box_cost.min(axis=-1).reshape(-1, 1),
            self.item_assign_box_cost.mean(axis=-1).reshape(-1, 1),
            self.item_cur_box.reshape(-1, 1),
        ], axis=-1)
        cols = ['migrationCost']+resource_types+['amax', 'amin', 'amean', 'inbox']
        items_pd = pd.DataFrame(items_state, columns=cols)
        ii, self.dense_items_idxes, self.items_map = 0, np.ones(len(items_state), dtype=int) * -1, {}
        for _, g in items_pd.groupby(by=cols):
            self.items_map[ii] = g.index.tolist()
            self.dense_items_idxes[g.index.tolist()] = ii
            ii += 1
        
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + resource_types].values
        self.box_remain_resources = self.box_infos[resource_types].values - self.init_placement.T.dot(self.item_infos[resource_types].values)
        self.box_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        if self.len_numa_res > 0:
            # self.numa_remain_resources = self.box_cur_numa - self.cur_placement.T.dot(self.item_cur_numa)
            self.numa_remain_resources = self.box_cur_numa - self.init_placement.T.dot(self.init_items_numa)
            self.numa_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0

        assert (self.box_remain_resources>=0).all(), f"Box Resource Constraints Unsatisfied! Index: {np.where(self.box_remain_resources<0)}"
        self.box_cur_state[:, -len(resource_types):] = self.box_remain_resources
        
        fixed_placement = self.init_placement.copy()
        fixed_placement[((self.item_infos['count'].values <= 0) | (self.item_infos['canMigrate'].values <= 0)).astype(int) <= 0] = 0
        self.box_fixed_remain_resources = self.box_infos[resource_types].values.astype(float) - fixed_placement.T.dot(self.item_infos[resource_types].values).astype(float)
        self.box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = self.box_infos[resource_types].values.max(axis=0) * 2
        # pdb.set_trace()
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources, 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(self.box_cur_numa), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(self.box_cur_numa), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
       
        self.actions_available = ((self.cur_placement + self.item_mutex_box == 0) & resource_enough & numa_resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        
        self.invalid_action_count, self.move_count = 0, 0

        self._init_reward_scale()


        self.current_step = 0
        self.bad_done = False
        self.done_rewards = 0
        
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


        self.edges = np.expand_dims(np.arange(self.num_items + self.num_boxes), axis=-1)
        
        for item_id in range(len(self.item_cur_state)):
        # for item_id in range(len(self.item_infos)):
            box_id = self.item_cur_box[item_id]
            self.edges[item_id + self.num_boxes, 0] = box_id

        return self._get_dict_state()

    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)

    def find_numa_placement(self, action, numa_action):
        item_i, box_j = action
        if numa_action is not None:
            numa_action = np.array(numa_action)
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[action[0]]).reshape(-1)
            if (self.box_cur_state[action[1], 1] <= 0) and (self.numa_remain_resources[action[1]] < item_i_numa).any():
                numa_action = None
        numa_action = self.stored_numa_actions.get(tuple(action)) if numa_action is None else numa_action
        
        if (numa_action is None) and (self.len_numa_res > 0):
            numa_action = np.zeros(self.max_box_numa, dtype=int)
            if self.box_cur_state[box_j, 1] >= 1:
                numa_action[:self.item_numa_num[item_i]] = 1
            else:
                box_j_numa = self.numa_remain_resources[box_j].reshape(self.max_box_numa, self.len_numa_res)
                item_i_numa = dcp(self.item_numa_res[item_i])
                numa_idxes = np.where((box_j_numa >= item_i_numa).all(axis=-1))[0]
                # if len(numa_idxes) <= 0:
                #     numa_idxes = np.arange(self.max_box_numa)
                # numa_idxes = np.arange(self.max_box_numa)
                # selected_numa = numa_idxes[np.argsort((item_i_numa / (box_j_numa + 1)).mean(axis=-1)[numa_idxes])[:self.item_numa_num[item_i]]]
                item_i_numa[item_i_numa==0] = 1
                selected_numa = numa_idxes[np.argsort(((box_j_numa % item_i_numa) / item_i_numa).mean(axis=-1)[numa_idxes])][:self.item_numa_num[item_i]]
                if len(selected_numa) < self.item_numa_num[item_i]:
                    selected_numa = list(set(list(range(self.max_box_numa))) - set(selected_numa))[:self.item_numa_num[item_i]-len(selected_numa)] + list(selected_numa)
                    numa_action[selected_numa] = 1
                else:
                    numa_action[selected_numa] = 1
        
        if (numa_action is not None):
            if (len(numa_action) < self.max_box_numa):
                numa_action_t = np.zeros(self.max_box_numa, dtype=int)
                numa_action_t[:len(numa_action)] = numa_action
                numa_action = numa_action_t
            numa_action = np.array(numa_action)
        
        return numa_action

    def step(self, action, numa_action=None):
        self.bad_done = False

        action = action.tolist()
        numa_action = self.find_numa_placement(action, numa_action)
        action_available, action_info = self.is_action_available(action, numa_action)

        self.invalid_action_count += int(not action_available)
        self.move_count += 1

        if action_available:
            
            costs = self._get_costs(action)
            reward = self._get_reward(costs)
            self._update_state(action, numa_action)

        else:
            # print("test: invalid")
            costs, reward = {'migration_cost': 0, 'box_used_cost': 0, 'item_assign_cost': 0}, invalid_action_punish

        state = self._get_dict_state()
        done = self._is_done()

        self.done_rewards -= costs['migration_cost'] + costs['item_assign_cost'] + costs['box_used_cost']
        
        # self.done_rewards += fragmentation_reward

        self.edges[action[0] + self.num_boxes] = action[1]

        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
            'numa_action': numa_action,
        }
        infos.update(costs)
        if done:
            done_score = {'done_score': self.get_cur_state_score()}
            infos.update(done_score)
            done_rewards = {'done_rewards': self.done_rewards}
            infos.update(done_rewards)
            if self.mode == 'train' or self.mode == 'eval' or self.mode == 'test':
                self.index -= 1
                self.reset()
        self.current_step += 1 

        return state, reward, done, infos
    
    def get_item_cur_box(self, item_i):
        return self.item_cur_box[item_i]
    
    def _update_box_action_available(self, box_list):
        resource_enough = (np.expand_dims(self.box_cur_state[box_list, -len(resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True

        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources[box_list], 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(box_list), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(box_list), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
            
        self.actions_available[:, box_list] = ((self.cur_placement[:, box_list] + self.item_mutex_box[:, box_list] == 0) & resource_enough & numa_resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
    
    def _update_state(self, action, numa_action):
        if numa_action is not None:
            numa_action = np.array(numa_action)
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
        if numa_action is not None:
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
        else:
            item_i_numa = None
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
                if numa_action is not None:
                    self.numa_remain_resources[box_j] += self.item_cur_numa[item_i]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
                if numa_action is not None:
                    self.numa_remain_resources[box_k] -= item_i_numa
                if self.item_cur_state[item_i, 0] <= 1:
                    self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        elif self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
            if numa_action is not None:
                self.numa_remain_resources[box_k] -= item_i_numa
            self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(resource_types):]
        if (self.len_numa_res > 0) and (numa_action is not None):
            self.item_new_numa[(item_i, box_k)] = item_i_numa
            self.stored_numa_actions[(item_i, box_k)] = numa_action
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] -= 1

        self._update_box_action_available([box_j, box_k])
    
    def get_vm_dense_mask(self, id=None):
        available_actions = self.get_available_actions()
        vm_mask = (available_actions.sum(axis=1) > 0).astype(bool)
        vm_can_be_selected_idxes = pd.DataFrame(self.dense_items_idxes[vm_mask], columns=['idx']).groupby(by=['idx']).apply(lambda x: x.index[0]).values
        vm_new_mask = np.zeros(vm_mask.astype(int).sum(), dtype=bool)
        vm_new_mask[vm_can_be_selected_idxes] = True
        vm_mask[vm_mask] = vm_new_mask
        expand_vm_mask = np.ones((self.max_num_items)).astype(bool)
        expand_vm_mask[:vm_mask.shape[0]] = ~vm_mask
        return expand_vm_mask
    
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


    def is_action_available(self, action, numa_action):
        if numa_action is not None:
            numa_action = np.array(numa_action)
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
        
        if numa_action is not None:
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
            to_numa = np.where(numa_action == 1)[0]
            if len(to_numa) != self.item_numa_num[item_i]:
                return False, f"Provided Numa Count {len(to_numa)} is not equal with Item Required Numa {self.item_numa_num[item_i]}!"
            if self.is_filter_unmovable:
                idx = self.filter_to_unfilter[item_i]
            else:
                idx = item_i
            from_numa = np.array(self.item_full_infos.iloc[idx]['numa'][0], dtype=int)
            # pdb.set_trace()
            numa_count = self.item_full_infos.iloc[idx]['numa'][1]
            if (self.box_cur_state[box_k, 1] <= 0) and (self.numa_remain_resources[box_k] < item_i_numa).any():
                return False, f"Box {box_k}'s remained numa resources {self.numa_remain_resources[box_k]} are not enough for Item {item_i} {item_i_numa}!"
        else:
            from_numa, to_numa, numa_count = [], [], 0
        
        # if self.actions_available[item_i, box_k] <= 0:
        #     return False, f"Action [{item_i}, {box_k}] is not available!"
        
        return True, f"Item {item_i} can move from Box {box_j} Numa {numa_count} {from_numa} to Box {box_k} Numa {to_numa}!"

    
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


    def calculate_fragmentation(self, box_states):
        # 获取每个PM的总资源容量
        total_memory = np.maximum(self.box_infos['memory'].values, 1e-8)  # 避免除零
        total_cpu = np.maximum(self.box_infos['cpu'].values, 1e-8)  # 避免除零
        
        # 计算每个PM的剩余资源比例
        memory_remaining_ratio = np.clip(box_states[:, 2] / total_memory, 0, 1)  # Memory is at index 2
        cpu_remaining_ratio = np.clip(box_states[:, 3] / total_cpu, 0, 1)  # CPU is at index 3
        # 计算每个PM的资源利用率（1 - 剩余比例）
        memory_utilization = 1 - memory_remaining_ratio
        cpu_utilization = 1 - cpu_remaining_ratio
        
        # 计算平均利用率
        avg_utilization = (memory_utilization + cpu_utilization) / 2

        # 碎片率：1 - 平均利用率（越低越好）
        fragmentation = 1 - np.mean(avg_utilization)
        return fragmentation

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
            # reward -= value
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

    def _get_dict_state(self, mode='max-min'):
        
        def safe_std(std_value):
            return std_value if std_value != 0 else 1

        vm_mean_migration_cost = np.mean(self.item_assign_box_cost, axis=1)
        vm_mean_std_migration_cost = safe_std(np.std(vm_mean_migration_cost))
        vm_max_migration_cost = np.max(self.item_assign_box_cost, axis=1)
        vm_max_std_migration_cost = safe_std(np.std(vm_max_migration_cost))
        vm_min_migration_cost = np.min(self.item_assign_box_cost, axis=1)
        vm_min_std_migration_cost = safe_std(np.std(vm_min_migration_cost))

        pm_mean_migration_cost = np.mean(self.item_assign_box_cost, axis=0)
        pm_mean_std_migration_cost = safe_std(np.std(pm_mean_migration_cost, axis=0))
        pm_max_migration_cost = np.max(self.item_assign_box_cost, axis=0)
        pm_max_std_migration_cost = safe_std(np.std(pm_max_migration_cost, axis=0))
        pm_min_migration_cost = np.min(self.item_assign_box_cost, axis=0)
        pm_min_std_migration_cost = safe_std(np.std(pm_min_migration_cost, axis=0))

        if mode == 'mean-std':
            pm_cpu_std = self.pm_cpu_std if self.pm_cpu_std != 0 else 1 
            pm_mem_std = self.pm_mem_std if self.pm_mem_std != 0 else 1
            pm_cpu_baseline_std = self.pm_cpu_baseline_std if self.pm_cpu_baseline_std != 0 else 1
            pm_vm_count_std = self.pm_vm_count_std if self.pm_vm_count_std != 0 else 1
            pm_vport_std = self.pm_vport_std if self.pm_vport_std != 0 else 1
            
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_mean) / pm_cpu_std
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_mean) / pm_mem_std
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_mean) / pm_cpu_baseline_std
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_mean) / pm_vm_count_std
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_mean) / pm_vport_std
            
            pm_mean_migration_cost_norm = (pm_mean_migration_cost - np.mean(pm_mean_migration_cost)) / pm_mean_std_migration_cost
            pm_max_migration_cost_norm = (pm_max_migration_cost - np.mean(pm_max_migration_cost)) / pm_max_std_migration_cost
            pm_min_migration_cost_norm = (pm_min_migration_cost - np.mean(pm_min_migration_cost)) / pm_min_std_migration_cost

            pm_info_norm = np.column_stack([self.box_cur_state[:, :2], pm_mean_migration_cost_norm, pm_max_migration_cost_norm, pm_min_migration_cost_norm,
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
                                            pm_vm_count_norm, pm_vport_norm])

            # pm_info_norm = np.column_stack([self.box_cur_state[:, :2],
            #                                 pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
            #                                 pm_vm_count_norm, pm_vport_norm])

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
            
            vm_mean_migration_cost_norm = (vm_mean_migration_cost - np.mean(vm_mean_migration_cost)) / vm_mean_std_migration_cost
            vm_max_migration_cost_norm = (vm_max_migration_cost - np.mean(vm_max_migration_cost)) / vm_max_std_migration_cost
            vm_min_migration_cost_norm = (vm_min_migration_cost - np.mean(vm_min_migration_cost)) / vm_min_std_migration_cost
        
            vm_info_norm = np.column_stack([self.item_cur_state[:, :4], vm_mean_migration_cost_norm, vm_max_migration_cost_norm, vm_min_migration_cost_norm,
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
                                            vm_vm_count_norm, vm_vport_norm])

            # vm_info_norm = np.column_stack([self.item_cur_state[:, :4],
            #                                 vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
            #                                 vm_vm_count_norm, vm_vport_norm])

        elif mode == 'max-min':
            pm_cpu_norm = (self.box_cur_state[:, 2] - self.pm_cpu_min) / (self.pm_cpu_max - self.pm_cpu_min) if self.pm_cpu_max != self.pm_cpu_min else np.zeros(len(self.box_cur_state))
            pm_mem_norm = (self.box_cur_state[:, 3] - self.pm_mem_min) / (self.pm_mem_max - self.pm_mem_min) if self.pm_mem_max != self.pm_mem_min else np.zeros(len(self.box_cur_state))
            pm_cpu_baseline_norm = (self.box_cur_state[:, 4] - self.pm_cpu_baseline_min) / (self.pm_cpu_baseline_max - self.pm_cpu_baseline_min) if self.pm_cpu_baseline_max != self.pm_cpu_baseline_min else np.zeros(len(self.box_cur_state))
            pm_vm_count_norm = (self.box_cur_state[:, 5] - self.pm_vm_count_min) / (self.pm_vm_count_max - self.pm_vm_count_min) if self.pm_vm_count_max != self.pm_vm_count_min else np.zeros(len(self.box_cur_state))
            pm_vport_norm = (self.box_cur_state[:, 6] - self.pm_vport_min) / (self.pm_vport_max - self.pm_vport_min) if self.pm_vport_max != self.pm_vport_min else np.zeros(len(self.box_cur_state))

            
            pm_info_norm = np.column_stack([self.box_cur_state[:, :2], 
                                            pm_mean_migration_cost, pm_max_migration_cost, pm_min_migration_cost,
                                            pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm, pm_vm_count_norm, pm_vport_norm])

            # pm_info_norm = np.column_stack([self.box_cur_state[:, :2], 
            #                                 pm_cpu_norm, pm_mem_norm, pm_cpu_baseline_norm,
            #                                 pm_vm_count_norm, pm_vport_norm])


            vm_cpu_norm = (self.item_cur_state[:, 4] - self.vm_cpu_min) / (self.vm_cpu_max - self.vm_cpu_min) if self.vm_cpu_max != self.vm_cpu_min else np.zeros(len(self.item_cur_state))
            vm_mem_norm = (self.item_cur_state[:, 5] - self.vm_mem_min) / (self.vm_mem_max - self.vm_mem_min) if self.vm_mem_max != self.vm_mem_min else np.zeros(len(self.item_cur_state))
            vm_cpu_baseline_norm = (self.item_cur_state[:, 6] - self.vm_cpu_baseline_min) / (self.vm_cpu_baseline_max - self.vm_cpu_baseline_min) if self.vm_cpu_baseline_max != self.vm_cpu_baseline_min else np.zeros(len(self.item_cur_state))
            vm_vm_count_norm = (self.item_cur_state[:, 7] - self.vm_count_min) / (self.vm_count_max - self.vm_count_min) if self.vm_count_max != self.vm_count_min else np.zeros(len(self.item_cur_state))
            vm_vport_norm = (self.item_cur_state[:, 8] - self.vm_vport_min) / (self.vm_vport_max - self.vm_vport_min) if self.vm_vport_max != self.vm_vport_min else np.zeros(len(self.item_cur_state))
            
            vm_mean_migration_cost_norm = (vm_mean_migration_cost - np.min(vm_mean_migration_cost)) / (np.max(vm_mean_migration_cost) - np.min(vm_mean_migration_cost)) if np.max(vm_mean_migration_cost) != np.min(vm_mean_migration_cost) else np.zeros_like(vm_mean_migration_cost)
            vm_max_migration_cost_norm = (vm_max_migration_cost - np.min(vm_max_migration_cost)) / (np.max(vm_max_migration_cost) - np.min(vm_max_migration_cost)) if np.max(vm_max_migration_cost) != np.min(vm_max_migration_cost) else np.zeros_like(vm_max_migration_cost)
            vm_min_migration_cost_norm = (vm_min_migration_cost - np.min(vm_min_migration_cost)) / (np.max(vm_min_migration_cost) - np.min(vm_min_migration_cost)) if np.max(vm_min_migration_cost) != np.min(vm_min_migration_cost) else np.zeros_like(vm_min_migration_cost)


            vm_info_norm = np.column_stack([self.item_cur_state[:, :4], 
                                            vm_mean_migration_cost, vm_max_migration_cost, vm_min_migration_cost,
                                            vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm, vm_vm_count_norm, vm_vport_norm])

            # vm_info_norm = np.column_stack([self.item_cur_state[:, :4],
            #                                 vm_cpu_norm, vm_mem_norm, vm_cpu_baseline_norm,
            #                                 vm_vm_count_norm, vm_vport_norm])


            # 获取每个item对应的box索引
            item_box_indices = self.item_cur_box

            # 计算每个box上的item状态平均值
            box_item_avg = np.zeros((len(self.box_infos), vm_info_norm.shape[1]))
            for box_idx in range(len(self.box_infos)):
                items_in_box = vm_info_norm[item_box_indices == box_idx]
                if len(items_in_box) > 0:
                    box_item_avg[box_idx] = np.mean(items_in_box, axis=0)

            # 为每个box添加其上item的平均状态
            pm_info_with_item_avg = np.column_stack([
                pm_info_norm,
                box_item_avg,
            ])

            # 为每个item添加其对应box的状态
            vm_info_with_box = np.column_stack([
                vm_info_norm,
                pm_info_with_item_avg[item_box_indices]
            ])

        expand_vm_info_norm = np.zeros((self.max_num_items, vm_info_norm.shape[1]))
        expand_pm_info_norm = np.zeros((self.max_num_boxes, pm_info_norm.shape[1]))
        expand_vm_info_norm[:vm_info_norm.shape[0], :] = vm_info_norm
        expand_pm_info_norm[:pm_info_norm.shape[0], :] = pm_info_norm

        state = {'vm_info': expand_vm_info_norm,
                'pm_info': expand_pm_info_norm,
                'num_steps': self.current_step,
                'num_vms': len(self.item_cur_state),
                'edges': self.edges
                }
        return state

    def _is_done(self):
        done = False
        if self.invalid_action_count >= invalid_action_done_count:
            done = True
        if self.is_limited_count and (self.move_count >= self.configs['maxMigration']):
        # if self.is_limited_count and (self.move_count >= 100):
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
        if self.is_filter_unmovable:
            cur_placement = np.array(self.init_placement).copy()
            cur_placement[self.item_init_movable] = self.cur_placement.copy()
        else:
            cur_placement = self.cur_placement.copy()
        used_costs = self.box_cur_state[cur_placement.sum(axis=0)>0, 0].sum()
        assign_costs = (self.item_assign_box_cost_origin * cur_placement).sum()
        return used_costs + assign_costs



if __name__ == '__main__':
    data_path = '../data/nocons_finite_easy/8621'
    s = time.time()
    env = ECSEnvironment(data_path=data_path)
    print(time.time() - s)
