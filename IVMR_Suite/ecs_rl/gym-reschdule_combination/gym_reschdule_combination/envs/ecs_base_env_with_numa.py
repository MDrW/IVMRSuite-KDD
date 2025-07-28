import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import random
import os
from ecs_data_process.data_config import *
import json
import time
from scipy import sparse
from copy import deepcopy as dcp
import pdb


class ECSEnvironment:
    def __init__(self, data_path, is_limited_count=True, is_filter_unmovable=False, is_process_numa=False):
        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.item_full_infos = pd.read_pickle(os.path.join(data_path, 'item_infos.pkl'))
        self.box_full_infos = pd.read_pickle(os.path.join(data_path, 'box_infos.pkl'))
        
        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost_origin = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box_origin = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()
        # self.item_mix_item_cost = sparse.load_npz(os.path.join(data_path, 'item_mix_item_cost.npz')).toarray()
        # self.item_mutex_item = sparse.load_npz(os.path.join(data_path, 'item_mutex_item.npz')).toarray()
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)
        self.is_limited_count = is_limited_count
        self.is_filter_unmovable = is_filter_unmovable
        
        if (self.configs['maxBoxNuma'] > 0) and (self.configs['maxItemNuma'] > 0) and is_process_numa:
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
        
        self.seed(0)
        self.reset()
    
    def reset(self):
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
        
        return self._get_state()
    
    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)
        
    def step_ignore_resource_satisfaction(self, action, numa_action=None):
        if (numa_action is None) and (self.len_numa_res > 0):
            numa_action = self.find_numa_placement(action)
        if (numa_action is not None):
            if (len(numa_action) < self.max_box_numa):
                numa_action_t = np.zeros(self.max_box_numa, dtype=int)
                numa_action_t[:len(numa_action)] = numa_action
                numa_action = numa_action_t
            numa_action = np.array(numa_action)
        costs = self._get_costs(action)
        reward = self._get_reward(costs)
        self._update_state(action, numa_action)
        unsatisfied_boxes = np.where((self.box_cur_state[:, -len(resource_types):] < 0).any(axis=-1) & (self.box_cur_state[:, 1] <= 0) & (self.numa_remain_resources < 0).any(axis=-1))[0]
        state, done = None, False
        infos = {'unsatisfied_boxes': unsatisfied_boxes}
        return state, reward, done, infos
        
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
        numa_action = self.find_numa_placement(action, numa_action)
        
        action_available, action_info = self.is_action_available(action, numa_action)
        self.invalid_action_count += int(not action_available)
        self.move_count += 1
        if action_available:
            costs = self._get_costs(action)
            reward = self._get_reward(costs)
            self._update_state(action, numa_action)
        else:
            costs, reward = {'migration_cost': 1e10, 'box_used_cost': 1e10, 'item_assign_cost': 1e10}, invalid_action_punish
        
        state = self._get_state()
        done = self._is_done()
        
        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
            'numa_action': numa_action,
        }
        infos.update(costs)
        
        return state, reward, done, infos
    
    # def step_ignore_resource_satisfaction(self, action):
    #     costs = self._get_costs(action)
    #     reward = self._get_reward(costs)
    #     self._update_state(action)
    #     unsatisfied_boxes = np.where((self.box_cur_state[:, -len(resource_types):] < 0).any(axis=-1) & (self.box_cur_state[:, 1] <= 0))[0]
    #     state, done = None, False
    #     infos = {'unsatisfied_boxes': unsatisfied_boxes}
    #     return state, reward, done, infos
    
    def undo_step(self, action):
        action_available, action_info = self.is_undo_action_available(action)
        self.move_count -= 1
        if action_available:
            costs = self._get_undo_costs(action)
            reward = self._get_reward(costs)
            self._undo_update_state(action)
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
    
    def get_items_cur_numa(self):
        if self.len_numa_res > 0:
            item_cur_numa_res = dcp(self.item_cur_numa)
            for k, v in self.item_new_numa.items():
                item_cur_numa_res[k[0]] = v
            numa_actions = (item_cur_numa_res.reshape(len(item_cur_numa_res), self.max_box_numa, self.len_numa_res) > 0).any(axis=-1).astype(int)
        else:
            numa_actions = np.zeros((len(self.item_cur_state), self.max_box_numa), dtype=int)
        items, boxes = np.where(self.cur_placement == 1)
        items_cur_numa = np.zeros((len(self.item_cur_state), len(self.box_cur_state), self.max_box_numa), dtype=int)
        for i in range(len(items)):
            items_cur_numa[items[i], boxes[i]] = numa_actions[i]
        return items_cur_numa
    
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
        item_i_numa = self.item_new_numa.get((item_i, box_j))
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
                if (self.len_numa_res > 0) and (item_i_numa is not None):
                    self.numa_remain_resources[box_j] += self.item_new_numa[(item_i, box_j)]
                if self.item_cur_state[item_i, 0] <= 0:
                    self.box_fixed_remain_resources[box_j] += self.item_cur_state[item_i, -len(resource_types):]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(resource_types):] -= self.item_cur_state[item_i, -len(resource_types):]
                if (self.len_numa_res > 0) and (item_i_numa is not None):
                    self.numa_remain_resources[box_k] -= self.item_cur_numa[item_i]
        elif self.box_cur_state[box_j, 1] <= 0:
            self.box_cur_state[box_j, -len(resource_types):] += self.item_cur_state[item_i, -len(resource_types):]
            if (self.len_numa_res > 0) and (item_i_numa is not None):
                self.numa_remain_resources[box_j] += self.item_new_numa[(item_i, box_j)]
            self.box_fixed_remain_resources[box_j] += self.item_cur_state[item_i, -len(resource_types):]
        
        if (self.len_numa_res > 0) and (item_i_numa is not None):
            del self.item_new_numa[(item_i, box_j)]
            del self.stored_numa_actions[(item_i, box_j)]
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] += 1

        self._update_box_action_available([box_j, box_k])
    
    # def get_stored_numa_actions(self):
    #     return self.stored_numa_actions
    
    def action_space(self):
        item_actions = list(range(len(self.item_cur_state)))
        box_actions = list(range(len(self.box_cur_state)))
        return [item_actions, box_actions]
    
    def state_space(self):
        return [len(x) for x in self._get_state()]
    
    def get_available_actions(self):
        return self.actions_available
    
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
        
        if (self.len_numa_res > 0):
            # pdb.set_trace()
            from_numa = np.where(((self.item_new_numa[(item_i, box_j)].reshape(self.max_box_numa, self.len_numa_res)) > 0).any(axis=-1))[0]
            if self.is_filter_unmovable:
                idx = self.filter_to_unfilter[item_i]
            else:
                idx = item_i
            to_numa = np.array(self.item_full_infos.iloc[idx]['numa'][0], dtype=int)
            if len(to_numa) != self.item_numa_num[item_i]:
                return False, f"Provided Numa Count {len(to_numa)} is not equal with Item {item_i} Required Numa {self.item_numa_num[item_i]}!"
            numa_count = self.item_full_infos.iloc[idx]['numa'][1]
            if (self.box_cur_state[box_k, 1] <= 0) and (self.numa_remain_resources[box_k] < self.item_cur_numa[item_i]).any():
                return False, f"Box {box_k}'s remained numa resources {self.numa_remain_resources[box_k]} are not enough for Item {item_i} {self.item_cur_numa[item_i]}!"
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
    
    def _get_reward(self, costs):
        reward = 0
        for k, v in costs.items():
            reward -= v
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
    
    def _is_done(self):
        done = False
        if self.invalid_action_count >= invalid_action_done_count:
            done = True
        if self.is_limited_count and (self.move_count >= self.configs['maxMigration']):
            done = True
        if self.actions_available.sum() <= 0:
            done = True
        
        return done
    
    def render(self, mode='human'):
        pass
    
    def eval_move_sequence(self, action_list, numa_action_list=None):
        stored_numa_actions = dcp(self.stored_numa_actions)
        self.reset()
        self.stored_numa_actions = stored_numa_actions
        total_reward = 0
        costs = {'migration_cost': 0, 'item_assign_cost': 0, 'box_used_cost': 0}
        invalid_num = 0
        for ai, action in enumerate(action_list):
            # available, info = self.is_action_available(action)
            available = bool(self.get_available_actions()[action[0], action[1]])
            if available:
                if (numa_action_list is not None) and (len(numa_action_list) == len(action_list)):
                    numa_action = numa_action_list[ai]
                else:
                    numa_action = None
                _, reward, done, infos = self.step(action, numa_action)
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
    
    def get_cur_state_score(self, is_contain_migration=False):
        if self.is_filter_unmovable:
            cur_placement = np.array(self.init_placement).copy()
            cur_placement[self.item_init_movable] = self.cur_placement.copy()
        else:
            cur_placement = self.cur_placement.copy()
        used_costs = self.box_cur_state[cur_placement.sum(axis=0)>0, 0].sum()
        assign_costs = (self.item_assign_box_cost_origin * cur_placement).sum()
        if is_contain_migration:
            migration_cost = self.item_infos[(self.init_placement != cur_placement).any(axis=-1)]['migrationCost'].sum()
        else:
            migration_cost = 0
        return used_costs + assign_costs + migration_cost


if __name__ == '__main__':
    
    data_path = '../data/nocons_finite_easy/8621'
    env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True)
    env.reset()
    done = False
    total_reward = 0
    while not done:
        available = env.get_available_actions()
        # import pdb
        # pdb.set_trace()
        item = np.random.choice(np.where(available.sum(axis=1)>0)[0], size=1)[0]
        box = np.random.choice(np.where(available[item]>0)[0], size=1)[0]
        action = [item, box]
        s, r, done, info = env.step(action)
        total_reward += r
        print(f"Step {info['move_count']}: reward {r}, done {done}, {info['action_info']}.")
    print(total_reward)
