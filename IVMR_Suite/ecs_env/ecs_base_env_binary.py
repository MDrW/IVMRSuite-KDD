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
        self.item_mix_item_cost_origin = sparse.load_npz(os.path.join(data_path, 'item_mix_item_cost.npz')).toarray()
        self.item_mutex_item_origin = sparse.load_npz(os.path.join(data_path, 'item_mutex_item.npz')).toarray()
        
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
                if item_i_numa[1] > self.max_box_numa:
                    self.item_infos.loc[item_i, 'canMigrate'] = 0
                    self.item_full_infos.loc[item_i, 'canMigrate'] = 0
                if len(item_i_numa[0]) < item_i_numa[1]:
                    item_i_numa[0] = list(range(item_i_numa[1]))
                elif len(item_i_numa[0]) > item_i_numa[1]:
                    item_i_numa[0] = item_i_numa[0][:item_i_numa[1]]
                for idx in item_i_numa[0][:self.max_box_numa]:
                    self.init_items_numa[item_i][int(idx)*self.len_numa_res:(int(idx)+1)*self.len_numa_res] = item_i_numa[2:]
                self.init_items_numa_num[item_i] = item_i_numa[1]
                self.init_items_numa_res[item_i] = item_i_numa[2:]
        else:
            self.len_numa_res, self.max_box_numa, self.init_boxes_numa, self.init_items_numa, self.init_items_numa_num, self.init_items_numa_res = 0, 0, None, None, None, None
        
        self.seed(0)
        self.reset()
    
    def reset(self):
        # item_cur_state: max_move_count, is_infinite, canMigrate
        self.item_assign_box_cost, self.item_mutex_box = self.item_assign_box_cost_origin.copy(), self.item_mutex_box_origin.copy()
        self.item_mix_item_cost, self.item_mutex_item = self.item_mix_item_cost_origin.copy(), self.item_mutex_item_origin.copy()
        self.cur_placement = np.array(self.init_placement).copy()
        self.item_cur_state = self.item_infos[['count', 'isInfinite', 'canMigrate', 'migrationCost'] + self.configs["resource_types"]].values
        # self.item_cur_state[:, 0] = max_item_move_count
        self.item_init_movable = (self.item_cur_state[:, 0] > 0) & (self.item_cur_state[:, 2] > 0)
        
        self.item_mutex_item[range(len(self.item_mutex_item)), range(len(self.item_mutex_item))] = 0
        self.item_mix_item_cost[range(len(self.item_mix_item_cost)), range(len(self.item_mix_item_cost))] = 0
        self.item_mutex_item_idxes = np.zeros(len(self.item_mutex_item), dtype=bool)
        self.item_mutex_item_idxes[sparse.coo_matrix(self.item_mutex_item).row] = True
        self.item_mutex_item_idxes[sparse.coo_matrix(self.item_mutex_item).col] = True
        self.item_mutex_item_flag = self.item_mutex_item_idxes.any()
        self.item_mix_idxes = np.zeros(len(self.item_mix_item_cost), dtype=bool)
        self.item_mix_idxes[sparse.coo_matrix(self.item_mix_item_cost).row] = True
        self.item_mix_idxes[sparse.coo_matrix(self.item_mix_item_cost).col] = True
        self.item_mix_flag = self.item_mix_idxes.any()
        
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
            self.additional_item_mutex_box = np.zeros_like(self.item_mutex_box, dtype=int)
            refine_flag = self.item_mutex_item_flag & (~self.item_init_movable)
            if refine_flag.any():
                self.additional_item_mutex_box[self.item_mutex_item_idxes] = np.matmul(self.item_mutex_item[self.item_mutex_item_idxes][:, refine_flag], self.cur_placement[refine_flag])
            self.additional_item_assign_box = np.zeros_like(self.item_assign_box_cost, dtype=float)
            refine_flag = self.item_mix_flag & (~self.item_init_movable)
            if refine_flag.any():
                self.additional_item_assign_box[self.item_mix_idxes] = np.matmul(self.item_mix_item_cost[self.item_mix_idxes][:, refine_flag], self.cur_placement[refine_flag])
            self.item_cur_state = self.item_cur_state[self.item_init_movable]
            self.item_cur_box = self.item_cur_box[self.item_init_movable]
            self.item_assign_box_cost = (self.item_assign_box_cost + self.additional_item_assign_box)[self.item_init_movable]
            self.item_mutex_box = ((self.item_mutex_box + self.additional_item_mutex_box) > 0).astype(int)[self.item_init_movable]
            self.cur_placement = self.cur_placement[self.item_init_movable]
            self.item_mix_item_cost = self.item_mix_item_cost[self.item_init_movable][:, self.item_init_movable]
            self.item_mutex_item = self.item_mutex_item[self.item_init_movable][:, self.item_init_movable]
            self.item_mutex_item_idxes = self.item_mutex_item_idxes[self.item_init_movable]
            self.item_mix_idxes = self.item_mix_idxes[self.item_init_movable]
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
        else:
            self.init_idxes_map = dict(zip(range(len(self.item_cur_state)), range(len(self.item_cur_state))))
            self.filter_to_unfilter = dict(zip(range(len(self.item_cur_state)), range(len(self.item_cur_state))))
        
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + self.configs["resource_types"]].values
        self.box_remain_resources = self.box_infos[self.configs["resource_types"]].values - self.init_placement.T.dot(self.item_infos[self.configs["resource_types"]].values)
        self.box_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        if self.len_numa_res > 0:
            # self.numa_remain_resources = self.box_cur_numa - self.cur_placement.T.dot(self.item_cur_numa)
            self.numa_remain_resources = self.box_cur_numa - self.init_placement.T.dot(self.init_items_numa)
            self.numa_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        # assert (self.box_remain_resources>=0).all(), f"Box Resource Constraints Unsatisfied! Index: {np.where(self.box_remain_resources<0)}"
        self.box_cur_state[:, -len(self.configs["resource_types"]):] = self.box_remain_resources
        
        fixed_placement = self.init_placement.copy()
        fixed_placement[((self.item_infos['count'].values <= 0) | (self.item_infos['canMigrate'].values <= 0)).astype(int) <= 0] = 0
        self.box_fixed_remain_resources = self.box_infos[self.configs["resource_types"]].values.astype(float) - fixed_placement.T.dot(self.item_infos[self.configs["resource_types"]].values).astype(float)
        self.box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = self.box_infos[self.configs["resource_types"]].values.max(axis=0) * 2
        # pdb.set_trace()
        # resource_enough = (np.expand_dims(self.box_cur_state[:, -len(self.configs["resource_types"]):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1)).all(axis=-1)
        resource_enough = ((np.expand_dims(self.box_cur_state[:, -len(self.configs["resource_types"]):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1)) | 
                           (np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1).repeat(len(self.box_cur_state), 1) <= 0)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources, 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(self.box_cur_numa), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(self.box_cur_numa), 1)), 2).repeat(self.max_box_numa, 2)
            # numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough = ((((box_numa_res >= item_numa_res) | (item_numa_res <= 0)).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
        
        # item_mutex_item_limit = (np.matmul(self.item_mutex_item, self.cur_placement) <= 0).astype(bool)
        # item_mutex_item_limit = ((sparse.csr_matrix(self.item_mutex_item).dot(sparse.csr_matrix(self.cur_placement))).toarray() <= 0).astype(bool)
        item_mutex_item_limit = np.ones_like(self.cur_placement, dtype=bool)
        if self.item_mutex_item_flag:
            item_mutex_item_limit[self.item_mutex_item_idxes] = (np.matmul(self.item_mutex_item[self.item_mutex_item_idxes][:, self.item_mutex_item_idxes], self.cur_placement[self.item_mutex_item_idxes]) <= 0).astype(bool)
        self.actions_available = ((self.cur_placement + self.item_mutex_box == 0) & item_mutex_item_limit & resource_enough & numa_resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        
        self.invalid_action_count, self.move_count = 0, 0
        
        # self.items_mix_cost = np.matmul(self.item_mix_item_cost, self.cur_placement)
        # self.items_mix_cost = sparse.csr_matrix(self.item_mix_item_cost).dot(sparse.csr_matrix(self.cur_placement)).toarray()
        self.items_mix_cost = np.ones_like(self.cur_placement, dtype=float)
        if self.item_mix_flag:
            self.items_mix_cost[self.item_mix_idxes] = np.matmul(self.item_mix_item_cost[self.item_mix_idxes][:, self.item_mix_idxes], self.cur_placement[self.item_mix_idxes])
        
        self._convert_constrs_vio_to_objective()
        
        return self._get_state()
    
    def _convert_constrs_vio_to_objective(self):
        # self.penalty_coef = self.item_infos['migrationCost'].max() + self.box_infos['cost'].max()
        # if len(self.item_assign_box_cost) > 0:
        #     self.penalty_coef += self.item_assign_box_cost.max() - self.item_assign_box_cost.min()
        # if len(self.item_mix_item_cost) > 0:
        #     self.penalty_coef += self.item_mix_item_cost.max() - self.item_mix_item_cost.min()
        # self.penalty_coef *= 10000
        # if (self.penalty_coef <= 0) or (self.configs['fixConflictOnly'] > 0):
        #     self.penalty_coef = 10000.
        self.penalty_coef = 1e33
        
        # Item Mutex Box Violation
        self.mutex_box_vio_cost = np.zeros_like(self.item_assign_box_cost, dtype=float)
        self.mutex_box_vio_cost[(self.cur_placement == 1) & (self.item_mutex_box == 1)] = self.penalty_coef
        self.mutex_box_num = (self.mutex_box_vio_cost > 0).sum()
        
        # Item Mutex Item Violation, Need to be updated when moving items
        item_mutex_item_vio = np.zeros(len(self.cur_placement), dtype=float)
        if self.item_mutex_item_flag:
            cur_placement = self.cur_placement[self.item_mutex_item_idxes]
            item_mutex_item_vio[self.item_mutex_item_idxes] = (self.item_mutex_item[self.item_mutex_item_idxes][:, self.item_mutex_item_idxes] * cur_placement[:, np.where(cur_placement==1)[1]]).sum(axis=-1)
        self.mutex_item_vio_cost = np.zeros_like(self.cur_placement, dtype=float)
        self.mutex_item_vio_cost[self.cur_placement == 1] = (item_mutex_item_vio * self.penalty_coef)
        self.mutex_item_num = (self.mutex_item_vio_cost > 0).sum()
        
        # Resources Violation, Need to be updated when moving items
        self.resource_vio_cost = np.zeros_like(self.cur_placement, dtype=float)
        self.item_resources = dcp(self.item_cur_state[:, -len(self.configs["resource_types"]):])
        if len(self.item_resources) > 0:
            self.item_resources = self.item_resources / np.clip(self.item_resources.mean(axis=0), 1e-5, np.inf)
            item_resources_vio = (self.item_resources * (self.box_remain_resources < 0).astype(int)[self.item_cur_box]).sum(axis=-1)
            self.resource_vio_cost[self.cur_placement == 1] = item_resources_vio * self.penalty_coef
            self.resource_vio_cost[:, self.box_cur_state[:, 1] >= 1] = 0
        self.res_vio_num = (self.box_remain_resources < 0).any(axis=-1).sum()
        
        # Numa Resources Violation, Need to be updated when moving items
        if self.len_numa_res > 0:
            self.item_numa_resources = dcp(self.item_cur_numa)
            self.numa_vio_cost = np.zeros_like(self.cur_placement, dtype=float)
            if len(self.item_numa_resources) > 0:
                self.item_numa_resources = self.item_numa_resources / np.clip(self.item_numa_resources.mean(axis=0), 1e-5, np.inf)
                item_numa_resources_vio = (self.item_numa_resources * (self.numa_remain_resources < 0).astype(int)[self.item_cur_box]).sum(axis=-1)
                self.numa_vio_cost[self.cur_placement > 0] = item_numa_resources_vio * self.penalty_coef
                self.numa_vio_cost[:, self.box_cur_state[:, 1] >= 1] = 0
            self.numa_vio_num = (self.numa_remain_resources < 0).any(axis=-1).sum()
            # pdb.set_trace()
        else:
            self.numa_vio_cost = np.zeros_like(self.resource_vio_cost, dtype=float)
            self.numa_vio_num = 0
        # pdb.set_trace()
        self.init_constrs_vio = (self.mutex_box_vio_cost > 0) | (self.mutex_item_vio_cost > 0) |\
            (((self.box_remain_resources < 0).any(axis=-1).astype(int)[self.item_cur_box].reshape(-1, 1) * self.cur_placement) > 0)
        if self.len_numa_res > 0:
            self.init_constrs_vio |= (((self.numa_remain_resources < 0).any(axis=-1).astype(int)[self.item_cur_box].reshape(-1, 1) * self.cur_placement) > 0)
    
    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)
        
    def step_ignore_resource_satisfaction(self, action, numa_action=None):
        numa_action = self.find_numa_placement(action, numa_action)
        
        costs = self._get_costs(action)
        # vio_costs = self._get_constrs_vio_costs(action[0], self.get_item_cur_box(action[0]), action[1])
        # costs.update(vio_costs)
        reward = self._get_reward(costs)
        
        self._update_state(action, numa_action)
        
        unsatisfied_flag = ((self.box_cur_state[:, -len(self.configs["resource_types"]):] < 0).any(axis=-1) & (self.box_cur_state[:, 1] <= 0))
        if self.len_numa_res > 0:
            unsatisfied_flag = unsatisfied_flag | ((self.numa_remain_resources < 0).any(axis=-1) & (self.box_cur_state[:, 1] <= 0))
        unsatisfied_boxes = np.where(unsatisfied_flag)[0]
        state, done = None, False
        infos = {'unsatisfied_boxes': unsatisfied_boxes}
        # if len(unsatisfied_boxes) > 0:
        #     pdb.set_trace()
        return state, reward, done, infos
        
    def find_numa_placement(self, action, numa_action):
        def _find_numa_action(action):
            item_i, box_j = action
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
            return numa_action
        
        if (numa_action is not None) and (self.len_numa_res > 0):
            numa_action = np.array(numa_action)
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[action[0]]).reshape(-1)
            if ((self.box_cur_state[action[1], 1] <= 0) and (self.numa_remain_resources[action[1]] < item_i_numa).any()) or ((numa_action.sum() != self.item_numa_num[action[0]])):
                numa_action = None
        numa_action = self.stored_numa_actions.get(tuple(action)) if numa_action is None else numa_action
        if (numa_action is None) and (self.len_numa_res > 0):
            numa_action = _find_numa_action(action)
        if (numa_action is not None) and (self.len_numa_res > 0):
            if (len(numa_action) < self.max_box_numa):
                numa_action_t = np.zeros(self.max_box_numa, dtype=int)
                numa_action_t[:len(numa_action)] = numa_action
                numa_action = numa_action_t
            numa_action = np.array(numa_action)
            if numa_action.sum() != self.item_numa_num[action[0]]:
                numa_action = None
        if self.len_numa_res <= 0:
            numa_action = None
        return numa_action
    
    def step(self, action, numa_action=None):
        numa_action = self.find_numa_placement(action, numa_action)
        
        action_available, action_info = self.is_action_available(action, numa_action)
        self.invalid_action_count += int(not action_available)
        self.move_count += 1
        if action_available:
            costs = self._get_costs(action)
            vio_costs = self._get_constrs_vio_costs(action[0], self.get_item_cur_box(action[0]), action[1])
            costs.update(vio_costs)
            reward = self._get_reward(costs)
            self._update_state(action, numa_action)
        else:
            costs = {'migration_cost': 1e99, 'box_used_cost': 1e99, 'item_assign_cost': 1e99, 'item_mix_cost': 1e99}
            vio_costs = {'mutex_box_vio': 1e99, 'mutex_item_vio': 1e99, 'resource_vio': 1e99, 'numa_vio': 1e99}
            costs.update(vio_costs)
            reward = -1e99
        
        state = self._get_state()
        done = self._is_done()
        self.mutex_box_num, self.mutex_item_num = (self.mutex_box_vio_cost > 0).sum(), (self.mutex_item_vio_cost > 0).sum()
        self.res_vio_num = (self.box_cur_state[:, -len(self.configs['resource_types']):] < 0).any(axis=-1).sum()
        if self.len_numa_res > 0:
            self.numa_vio_num = (self.numa_remain_resources < 0).any(axis=-1).sum()
        vio_info = f"Mutex Box: {self.mutex_box_num}, Mutex Item: {self.mutex_item_num}, Resource: {self.res_vio_num}, Numa: {self.numa_vio_num}!"
        
        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'vio_info': vio_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
            'numa_action': numa_action,
        }
        infos.update(costs)
        
        return state, reward, done, infos
    
    def undo_step(self, action):
        action_available, action_info = self.is_undo_action_available(action)
        self.move_count -= 1
        if action_available:
            costs = self._get_undo_costs(action)
            self._undo_update_state(action)
            vio_costs = self._get_constrs_vio_costs(action[0], action[1], action[2])
            costs.update(vio_costs)
            reward = self._get_reward(costs)
        else:
            costs = {'migration_cost': 1e99, 'box_used_cost': 1e99, 'item_assign_cost': 1e99, 'item_mix_cost': 1e99}
            vio_costs = {'mutex_box_vio': 1e99, 'mutex_item_vio': 1e99, 'resource_vio': 1e99, 'numa_vio': 1e99}
            costs.update(vio_costs)
            reward = -1e99
        
        state = self._get_state()
        done = self._is_done()
        self.mutex_box_num, self.mutex_item_num = (self.mutex_box_vio_cost > 0).sum(), (self.mutex_item_vio_cost > 0).sum()
        self.res_vio_num = (self.box_cur_state[:, -len(self.configs['resource_types']):] < 0).any(axis=-1).sum()
        if self.len_numa_res > 0:
            self.numa_vio_num = (self.numa_remain_resources < 0).any(axis=-1).sum()
        vio_info = f"Mutex Box: {self.mutex_box_num}, Mutex Item: {self.mutex_item_num}, Resource: {self.res_vio_num}, Numa: {self.numa_vio_num}!"
        
        infos = {
            'action_available': action_available,
            'action_info': action_info,
            'vio_info': vio_info,
            'move_count': self.move_count,
            'invalid_action_count': self.invalid_action_count,
        }
        infos.update(costs)
        
        return state, reward, done, infos
    
    def get_item_cur_box(self, item_i):
        return self.item_cur_box[item_i]
    
    def _update_box_action_available(self, box_list):
        # resource_enough = (np.expand_dims(self.box_cur_state[box_list, -len(self.configs["resource_types"]):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1)).all(axis=-1)
        resource_enough = ((np.expand_dims(self.box_cur_state[box_list, -len(self.configs["resource_types"]):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1)) | 
                           (np.expand_dims(self.item_cur_state[:, -len(self.configs["resource_types"]):], 1).repeat(len(box_list), 1) <= 0)).all(axis=-1)
        resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources[box_list], 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(box_list), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(box_list), 1)), 2).repeat(self.max_box_numa, 2)
            # numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough = ((((box_numa_res >= item_numa_res) | (item_numa_res <= 0)).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
        
        # item_mutex_item_limit = (np.matmul(self.item_mutex_item, self.cur_placement[:, box_list]) <= 0).astype(bool)
        # item_mutex_item_limit = (sparse.csr_matrix(self.item_mutex_item).dot(sparse.csr_matrix(self.cur_placement[:, box_list])).toarray() <= 0).astype(bool)
        item_mutex_item_limit = np.ones_like(self.cur_placement[:, box_list], dtype=bool)
        if self.item_mutex_item_flag:
            item_mutex_item_limit[self.item_mutex_item_idxes] = (np.matmul(self.item_mutex_item[self.item_mutex_item_idxes][:, self.item_mutex_item_idxes], self.cur_placement[self.item_mutex_item_idxes][:, box_list]) <= 0).astype(bool)
        self.actions_available[:, box_list] = ((self.cur_placement[:, box_list] + self.item_mutex_box[:, box_list] == 0) & item_mutex_item_limit & resource_enough & numa_resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
    
    def _update_constrs_vio(self, item_i, box_j, is_undo=False):
        # Item Mutex Box Violation
        if is_undo:
            self.mutex_box_vio_cost[item_i, box_j] = self.item_mutex_box[item_i, box_j] * self.penalty_coef
        else:
            self.mutex_box_vio_cost[item_i, box_j] = 0
        
        # Item Mutex Item Violation, Need to be updated when moving items
        item_mutex_item_vio = np.zeros(len(self.cur_placement), dtype=float)
        change_item_idxes = (self.item_mutex_item_idxes & (self.cur_placement[:, box_j] == 1))
        if change_item_idxes.any():
            cur_placement = self.cur_placement[change_item_idxes, box_j]
            item_mutex_item_vio[change_item_idxes] = (self.item_mutex_item[change_item_idxes][:, change_item_idxes] * cur_placement).sum(axis=-1)
        self.mutex_item_vio_cost[:, box_j] = (item_mutex_item_vio * self.penalty_coef)
        
        # Resources Violation, Need to be updated when moving items
        if self.box_cur_state[box_j, 1] <= 0:
            item_resources_vio = (self.item_resources * (self.box_cur_state[box_j, -len(self.configs['resource_types']):] < 0).astype(int)).mean(axis=-1)
            item_resources_vio[self.cur_placement[:, box_j] <= 0] = 0.
            self.resource_vio_cost[:, box_j] = item_resources_vio * self.penalty_coef
        
        # Numa Resources Violation, Need to be updated when moving items
        if (self.len_numa_res > 0) and (self.box_cur_state[box_j, 1] <= 0):
            item_numa_resources_vio = (self.item_numa_resources * (self.numa_remain_resources[box_j] < 0).astype(int)).mean(axis=-1)
            item_numa_resources_vio[self.cur_placement[:, box_j] <= 0] = 0.
            self.numa_vio_cost[:, box_j] = item_numa_resources_vio * self.penalty_coef
    
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
        if (self.len_numa_res > 0) and (numa_action is not None):
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
        else:
            item_i_numa = None
        if self.item_cur_state[item_i, 1] <= 0:
            if (self.box_cur_state[box_j, 1] <= 0):
                self.box_cur_state[box_j, -len(self.configs["resource_types"]):] += self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
                if (self.len_numa_res > 0) and (numa_action is not None):
                    self.numa_remain_resources[box_j] += self.item_cur_numa[item_i]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(self.configs["resource_types"]):] -= self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
                if (self.len_numa_res > 0) and (numa_action is not None):
                    self.numa_remain_resources[box_k] -= item_i_numa
                if self.item_cur_state[item_i, 0] <= 1:
                    self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
        elif self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(self.configs["resource_types"]):] -= self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
            if (self.len_numa_res > 0) and (numa_action is not None):
                self.numa_remain_resources[box_k] -= item_i_numa
            self.box_fixed_remain_resources[box_k] -= self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
        if (self.len_numa_res > 0) and (numa_action is not None):
            self.item_new_numa[(item_i, box_k)] = item_i_numa
            self.stored_numa_actions[(item_i, box_k)] = numa_action
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] -= 1

        self._update_box_action_available([box_j, box_k])
        self._update_constrs_vio(item_i, box_j, is_undo=False)
        if self.item_mix_flag:
            box_list = [box_j, box_k]
            self.items_mix_cost[self.item_mix_idxes][:, box_list] = np.matmul(self.item_mix_item_cost[self.item_mix_idxes][:, self.item_mix_idxes], self.cur_placement[self.item_mix_idxes][:, box_list])
    
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
                self.box_cur_state[box_j, -len(self.configs["resource_types"]):] += self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
                if (self.len_numa_res > 0) and (item_i_numa is not None):
                    self.numa_remain_resources[box_j] += self.item_new_numa[(item_i, box_j)]
                if self.item_cur_state[item_i, 0] <= 0:
                    self.box_fixed_remain_resources[box_j] += self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
            if (self.box_cur_state[box_k, 1] <= 0):
                self.box_cur_state[box_k, -len(self.configs["resource_types"]):] -= self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
                if (self.len_numa_res > 0) and (item_i_numa is not None):
                    self.numa_remain_resources[box_k] -= self.item_cur_numa[item_i]
        elif self.box_cur_state[box_j, 1] <= 0:
            self.box_cur_state[box_j, -len(self.configs["resource_types"]):] += self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
            if (self.len_numa_res > 0) and (item_i_numa is not None):
                self.numa_remain_resources[box_j] += self.item_new_numa[(item_i, box_j)]
            self.box_fixed_remain_resources[box_j] += self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
        
        if (self.len_numa_res > 0) and (item_i_numa is not None):
            del self.item_new_numa[(item_i, box_j)]
            del self.stored_numa_actions[(item_i, box_j)]
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] += 1

        self._update_box_action_available([box_j, box_k])
        self._update_constrs_vio(item_i, box_k, is_undo=True)
        if self.item_mix_flag:
            box_list = [box_j, box_k]
            self.items_mix_cost[self.item_mix_idxes][:, box_list] = np.matmul(self.item_mix_item_cost[self.item_mix_idxes][:, self.item_mix_idxes], self.cur_placement[self.item_mix_idxes][:, box_list])
    
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
            return False, f"Item {item_i} has been Moved!"
        
        # Item_i Current In Box_j?
        if self.cur_placement[item_i, box_j] <= 0:
            return False, f"Item {item_i} is not in Box {box_j}!"
        
        # Item_i Mutex Box_k?
        if self.item_mutex_box[item_i, box_k] >= 1:
            return False, f"Item {item_i} Mutex Box {box_k}!"
        
        # Item_i Mutex Item_j in Box_k?
        if ((self.item_mutex_item[item_i, :] * self.cur_placement[:, box_k]) > 0).any():
            return False, f"Item {item_i} Mutex Item in Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        item_res = self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(self.configs["resource_types"]):] < item_res)[item_res > 0].any():
            return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        if (self.len_numa_res > 0):
            if self.is_filter_unmovable:
                idx = self.filter_to_unfilter[item_i]
            else:
                idx = item_i
            from_numa = np.array(self.item_full_infos.iloc[idx]['numa'][0], dtype=int)
            # pdb.set_trace()
            numa_count = self.item_full_infos.iloc[idx]['numa'][1]
            if numa_count > 0 and (numa_action is None):
                return False, f"Provided Numa Count 0 is not equal with Item Required Numa {self.item_numa_num[item_i]}!"
            
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
            to_numa = np.where(numa_action == 1)[0]
            if len(to_numa) != self.item_numa_num[item_i]:
                return False, f"Provided Numa Count {len(to_numa)} is not equal with Item Required Numa {self.item_numa_num[item_i]}!"
            # if (self.box_cur_state[box_k, 1] <= 0) and (self.numa_remain_resources[box_k] < item_i_numa).any():
            box_k_numa = self.numa_remain_resources[box_k].reshape(-1, self.len_numa_res)
            num_can_place = ((box_k_numa >= self.item_numa_res[item_i]) | (self.item_numa_res[item_i].reshape(1, -1).repeat(len(box_k_numa), 0) <= 0)).all(axis=-1).sum()
            if (self.box_cur_state[box_k, 1] <= 0) and (num_can_place < self.item_numa_num[item_i]):
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
        
        if self.init_constrs_vio[item_i, box_k]:
            return True, f"Item {item_i} can move from Box {box_j} to Box {box_k} with Initial Constraints Violation!"
        
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
        
        # Item_i Mutex Item_j in Box_k?
        if ((self.item_mutex_item[item_i, :] * self.cur_placement[:, box_k]) > 0).any():
            return False, f"Item {item_i} Mutex Item in Box {box_k}!"
        
        # Box_k.Remain_Resource is enough for Item_i?
        item_res = self.item_cur_state[item_i, -len(self.configs["resource_types"]):]
        if (self.box_cur_state[box_k, 1] <= 0) and (self.box_cur_state[box_k, -len(self.configs["resource_types"]):] < item_res)[item_res > 0].any():
            return False, f"Box {box_k}'s remained resources are not enough for Item {item_i}!"
        
        if (self.len_numa_res > 0):
            # pdb.set_trace()
            if self.is_filter_unmovable:
                idx = self.filter_to_unfilter[item_i]
            else:
                idx = item_i
            
            numa_count = self.item_full_infos.iloc[idx]['numa'][1]
            if numa_count > 0:
                from_numa = np.where(((self.item_new_numa[(item_i, box_j)].reshape(self.max_box_numa, self.len_numa_res)) > 0).any(axis=-1))[0]
                to_numa = np.array(self.item_full_infos.iloc[idx]['numa'][0], dtype=int)
                if len(to_numa) != self.item_numa_num[item_i]:
                    return False, f"Provided Numa Count {len(to_numa)} is not equal with Item {item_i} Required Numa {self.item_numa_num[item_i]}!"
                if (self.box_cur_state[box_k, 1] <= 0) and (self.numa_remain_resources[box_k] < self.item_cur_numa[item_i])[self.item_cur_numa[item_i] > 0].any():
                # if (self.box_cur_state[box_k, 1] <= 0) and ((self.numa_remain_resources[box_k].reshape(-1, self.len_numa_res) >= self.item_numa_res[item_i]).all(axis=-1).sum() < self.item_numa_num[item_i]):
                    return False, f"Box {box_k}'s remained numa resources {self.numa_remain_resources[box_k]} are not enough for Item {item_i} {self.item_cur_numa[item_i]}!"
            else:
                from_numa, to_numa, numa_count = [], [], 0
        else:
            from_numa, to_numa, numa_count = [], [], 0
        
        # if self.actions_available[item_i, box_k] <= 0:
        #     return False, f"Action [{item_i}, {box_k}] is not available!"
        
        return True, f"Item {item_i} can move from Box {box_j} Numa {numa_count} {from_numa} to Box {box_k} Numa {to_numa}!"
    
    def _get_costs(self, action):
        item_i, box_k = action
        box_j = self.get_item_cur_box(item_i)
        
        migration_cost = self.item_cur_state[item_i, 3]
        
        is_box_k_not_used = int((self.cur_placement[:, box_k].sum() + self.infinite_item_cur_box[:, box_k].sum()) <= 0)
        box_used_cost = self.box_cur_state[box_k, 0] * is_box_k_not_used
        is_box_j_used = int((self.cur_placement[:, box_j].sum() + self.infinite_item_cur_box[:, box_j].sum()) == 1)
        box_used_cost -= self.box_cur_state[box_j, 0] * is_box_j_used
        
        if self.configs['fixConflictOnly'] > 0:
            item_assign_cost, item_mix_cost = 0, 0
        else:
            item_assign_cost = self.item_assign_box_cost[item_i, box_k] - self.item_assign_box_cost[item_i, box_j]
            
            items_in_box_j, items_in_box_k = self.cur_placement[:, box_j].copy(), self.cur_placement[:, box_k].copy()
            items_in_box_j[item_i] = 0
            item_mix_cost = (self.item_mix_item_cost[item_i, :] * items_in_box_k).sum() - (self.item_mix_item_cost[item_i, :] * items_in_box_j).sum()
        
        return {
            'migration_cost': migration_cost,
            'item_assign_cost': item_assign_cost,
            'box_used_cost': box_used_cost,
            'item_mix_cost': item_mix_cost,
        }
    
    def _get_constrs_vio_costs(self, item_i, box_j, box_k):
        mutex_box_vio_cost = self.mutex_box_vio_cost[item_i, box_k] - self.mutex_box_vio_cost[item_i, box_j]
        mutex_item_vio_cost = self.mutex_item_vio_cost[item_i, box_k] - self.mutex_item_vio_cost[item_i, box_j]
        resource_vio_cost = self.resource_vio_cost[item_i, box_k] - self.resource_vio_cost[item_i, box_j]
        numa_vio_cost = self.numa_vio_cost[item_i, box_k] - self.numa_vio_cost[item_i, box_j]
        
        return {
            'mutex_box_vio': mutex_box_vio_cost,
            'mutex_item_vio': mutex_item_vio_cost,
            'resource_vio': resource_vio_cost,
            'numa_vio': numa_vio_cost,
        }
        
    def _get_undo_costs(self, action):
        item_i, box_j, box_k = action
        
        migration_cost = -self.item_cur_state[item_i, 3]
        
        is_box_k_not_used = int((self.cur_placement[:, box_k].sum() + self.infinite_item_cur_box[:, box_k].sum()) <= 0)
        box_used_cost = self.box_cur_state[box_k, 0] * is_box_k_not_used
        is_box_j_used = int((self.cur_placement[:, box_j].sum() + self.infinite_item_cur_box[:, box_j].sum()) == 1)
        box_used_cost -= self.box_cur_state[box_j, 0] * is_box_j_used
        
        if self.configs['fixConflictOnly'] > 0:
            item_assign_cost, item_mix_cost = 0, 0
        else:
            item_assign_cost = self.item_assign_box_cost[item_i, box_k] - self.item_assign_box_cost[item_i, box_j]
            
            items_in_box_j, items_in_box_k = self.cur_placement[:, box_j].copy(), self.cur_placement[:, box_k].copy()
            items_in_box_j[item_i] = 0
            item_mix_cost = (self.item_mix_item_cost[item_i, :] * items_in_box_k).sum() - (self.item_mix_item_cost[item_i, :] * items_in_box_j).sum()
        
        return {
            'migration_cost': migration_cost,
            'item_assign_cost': item_assign_cost,
            'box_used_cost': box_used_cost,
            'item_mix_cost': item_mix_cost,
        }
    
    def _get_reward(self, costs):
        reward = 0
        for k, v in costs.items():
            reward -= v
        return reward
    
    def _get_reward_batch(self, box_list):
        migration_costs = self.item_cur_state[:, 3:4].repeat(len(box_list), axis=1)
        des_box_used_costs = self.box_cur_state[box_list, 0] * (self.cur_placement[:, box_list].sum(axis=0)<=0)
        origin_box_used_costs = (self.box_cur_state[:, 0] * (self.cur_placement.sum(axis=0)==1))[self.item_cur_box]
        box_used_costs = (des_box_used_costs[None].repeat(len(self.item_cur_state), axis=0) - origin_box_used_costs[:, None].repeat(len(box_list), axis=1))
        
        if self.configs['fixConflictOnly'] > 0:
            reward = -migration_costs - box_used_costs
        else:
            if self.item_mix_flag:
                self.items_mix_cost[self.item_mix_idxes][:, box_list] = np.matmul(self.item_mix_item_cost[self.item_mix_idxes][:, self.item_mix_idxes], self.cur_placement[self.item_mix_idxes][:, box_list])
            # self.items_mix_cost[:, box_list] = sparse.csr_matrix(self.item_mix_item_cost).dot(sparse.csr_matrix(self.cur_placement[:, box_list])).toarray()
            # item_mix_costs = self.items_mix_cost[:, box_list] - (self.items_mix_cost * self.cur_placement).sum(axis=1).reshape(-1, 1)
            item_mix_costs = self.items_mix_cost[:, box_list] - self.items_mix_cost[range(len(self.items_mix_cost)), self.item_cur_box].reshape(-1, 1)
            item_assign_costs = self.item_assign_box_cost[:, box_list] - (self.item_assign_box_cost * self.cur_placement).sum(axis=1).reshape(-1, 1)
            reward =  (- item_mix_costs - item_assign_costs - migration_costs - box_used_costs) # * self.actions_available[:, box_list]
        # pdb.set_trace()
        return reward
    
    def _get_vio_cost_batch(self):
        reward = (self.mutex_box_vio_cost * self.cur_placement).sum(axis=1).reshape(-1, 1) - self.mutex_box_vio_cost
        reward += (self.mutex_item_vio_cost * self.cur_placement).sum(axis=1).reshape(-1, 1) - self.mutex_item_vio_cost
        reward += (self.resource_vio_cost * self.cur_placement).sum(axis=1).reshape(-1, 1) - self.resource_vio_cost
        reward += (self.numa_vio_cost * self.cur_placement).sum(axis=1).reshape(-1, 1) - self.numa_vio_cost
        return reward
    
    def _get_state(self):
        state = [self.item_cur_state, self.box_cur_state, self.cur_placement, self.item_assign_box_cost, self.item_mutex_box]
        return state
    
    def _is_done(self):
        done = False
        if self.invalid_action_count >= 1:
            done = True
        if self.is_limited_count and (self.move_count >= self.configs['maxMigration']):
            done = True
        if self.actions_available.sum() <= 0:
            done = True
        if self.configs['fixConflictOnly'] > 0:
            if (self.mutex_box_num + self.mutex_item_num + self.res_vio_num + self.numa_vio_num) <= 0:
                done = True
            if (self.mutex_box_num + self.mutex_item_num + (self.resource_vio_cost != 0).sum() + (self.numa_vio_cost != 0).sum()) <= 0:
                done = True
        
        return done
    
    def render(self, mode='human'):
        pass
    
    def eval_move_sequence(self, action_list, numa_action_list=None):
        stored_numa_actions = dcp(self.stored_numa_actions)
        self.reset()
        self.stored_numa_actions = stored_numa_actions
        total_reward = 0
        costs = {'migration_cost': 0, 'item_assign_cost': 0, 'box_used_cost': 0, 'item_mix_cost': 0}
        vio_costs = {'mutex_box_vio': 0, 'mutex_item_vio': 0, 'resource_vio': 0, 'numa_vio': 0}
        costs.update(vio_costs)
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
        # mix_costs = (np.matmul(self.item_mix_item_cost_origin, cur_placement) * cur_placement).sum()
        mix_costs = (self.item_mix_item_cost_origin * cur_placement[:, np.where(cur_placement==1)[1]]).sum()
        # mix_costs = sparse.csr_matrix(self.item_mix_item_cost_origin).dot(sparse.csr_matrix(cur_placement)).sum()
        if is_contain_migration:
            migration_cost = self.item_infos[(self.init_placement != cur_placement).any(axis=-1)]['migrationCost'].sum()
        else:
            migration_cost = 0
        return used_costs + assign_costs + migration_cost + mix_costs
    
    def get_cur_state_vio_cost(self):
        vio_cost = (self.mutex_box_vio_cost * self.cur_placement).sum()
        vio_cost += (self.mutex_item_vio_cost * self.cur_placement).sum()
        vio_cost += (self.resource_vio_cost * self.cur_placement).sum()
        vio_cost += (self.numa_vio_cost * self.cur_placement).sum()
        # vio_num = {
        #     "MutexBox": {self.mutex_box_num}, 
        #     "MutexItem": {self.mutex_item_num}, 
        #     "Resource": {self.res_vio_num}, 
        #     "Numa": {self.numa_vio_num}
        # }
        vio_info = f"Mutex Box: {self.mutex_box_num}, Mutex Item: {self.mutex_item_num}, Resource: {self.res_vio_num}, Numa: {self.numa_vio_num}!"
        return vio_cost, vio_info


if __name__ == '__main__':
    start = time.time()
    data_path = '../data/ecs_data/1670'
    env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    env.reset()
    score = env.get_cur_state_score()
    vio, vio_info = env.get_cur_state_vio_cost()
    print(score, vio, vio_info)
    done = False or (env.get_available_actions().sum() <= 0)
    total_reward = 0
    end = time.time()
    print("Create and Reset Time:", end-start)
    while not done:
        s1 = time.time()
        available = env.get_available_actions()
        # import pdb
        # pdb.set_trace()
        item = np.random.choice(np.where(available.sum(axis=1)>0)[0], size=1)[0]
        box = np.random.choice(np.where(available[item]>0)[0], size=1)[0]
        action = [item, box]
        s, r, done, info = env.step(action)
        total_reward += r
        print(f"Step {info['move_count']}: reward {r}, done {done}, {info['action_info']} {time.time()-s1}s!")
    print(total_reward)
