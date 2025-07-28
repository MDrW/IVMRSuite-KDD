import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
import random
import os
import json
import time
from scipy import sparse
import pdb
from copy import deepcopy as dcp


class ECSGAEnvironment:
    def __init__(self, data_path, is_filter_unmovable=False, is_process_numa=False):
        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.item_full_infos = pd.read_pickle(os.path.join(data_path, 'item_infos.pkl'))
        self.box_full_infos = pd.read_pickle(os.path.join(data_path, 'box_infos.pkl'))
        
        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost_origin = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box_origin = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)
        self.resource_types = self.configs['resource_types']
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
        
        self.reset()
    
    def reset(self):
        self.item_assign_box_cost, self.item_mutex_box = self.item_assign_box_cost_origin.copy(), self.item_mutex_box_origin.copy()
        self.init_placement_copy = np.array(self.init_placement).copy()
        self.cur_placement = np.array(self.init_placement).copy()
        self.item_cur_state = self.item_infos[self.resource_types].values
        self.item_init_movable = (self.item_infos['canMigrate'] > 0).values.astype(bool)
        self.norm_item_migrations_costs = self.item_infos['migrationCost'].values
        self.norm_item_migrations_costs = self.norm_item_migrations_costs / (np.abs(self.norm_item_migrations_costs).max() + 1e-5)
        self.norm_item_assign_costs = ((self.item_assign_box_cost / (np.abs(self.item_assign_box_cost).max() + 1e-5))) * 100
        self.norm_box_used_costs = ((self.box_infos['cost']).values / ((np.abs(self.box_infos['cost'].values).max()) + 1e-5))
        
        self.item_cur_numa = dcp(self.init_items_numa)
        # self.item_new_numa, self.stored_numa_actions = {}, {}
        self.box_cur_numa = dcp(self.init_boxes_numa)
        self.item_numa_res = dcp(self.init_items_numa_res)
        self.item_numa_num = dcp(self.init_items_numa_num)
        
        if self.is_filter_unmovable:
            self.item_cur_state = self.item_cur_state[self.item_init_movable]
            self.item_assign_box_cost = self.item_assign_box_cost[self.item_init_movable]
            self.item_mutex_box = self.item_mutex_box[self.item_init_movable]
            self.cur_placement = self.cur_placement[self.item_init_movable]
            self.init_placement_copy = self.init_placement_copy[self.item_init_movable]
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
        
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + self.resource_types].values
        fixed_placement = self.init_placement.copy()
        fixed_placement[((self.item_infos['count'].values <= 0) | (self.item_infos['canMigrate'].values <= 0)).astype(int) <= 0] = 0
        box_fixed_remain_resources = self.box_infos[self.resource_types].values.astype(float) - fixed_placement.T.dot(self.item_infos[self.resource_types].values).astype(float)
        box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = self.box_infos[self.resource_types].values.max(axis=0) * 2
        self.box_cur_state[:, -len(self.resource_types):] = box_fixed_remain_resources
        
        if self.len_numa_res > 0:
            self.numa_remain_resources = self.box_cur_numa - self.init_placement.T.dot(self.init_items_numa)
            self.numa_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
            self.fixed_numa_resources = fixed_placement.T.dot(self.init_items_numa)
    
    def set_cur_placement(self, placement: np.array, items_numa_actions: np.array):
        self.cur_placement = placement.copy()
        self.cur_items_numa_actions = items_numa_actions.copy()
        box_remain_resources = self.box_cur_state[:, -len(self.resource_types):] - self.cur_placement.T.dot(self.item_cur_state).astype(float)
        box_remain_resources[self.box_cur_state[:, 1] > 0] = self.box_infos[self.resource_types].values.max(axis=0) * 2
        self.box_cur_state[:, -len(self.resource_types):] = box_remain_resources
        
        if self.len_numa_res > 0:
            item_cur_numa = np.zeros_like(self.item_cur_numa, dtype=float)
            items, boxes = np.where(placement == 1)
            for i in range(len(items)):
                numa_action = items_numa_actions[items[i], boxes[i]].reshape(-1, 1).repeat(self.len_numa_res, 1)
                item_i_numa = (numa_action * self.item_numa_res[items[i]]).reshape(-1)
                item_cur_numa[items[i]] = item_i_numa
            self.item_cur_numa = item_cur_numa
            self.numa_remain_resources = self.box_cur_numa - self.fixed_numa_resources - self.cur_placement.T.dot(self.item_cur_numa)
            self.numa_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(self.resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1] > 0] = True
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources, 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(self.box_cur_numa), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(self.box_cur_numa), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
        self.actions_available = ((self.cur_placement + self.item_mutex_box == 0) & resource_enough & numa_resource_enough).astype(int)
    
    def find_numa_placement(self, action):
        item_i, box_j = action
        numa_action = np.zeros(self.max_box_numa, dtype=int)
        if self.box_cur_state[box_j, 1] >= 1:
            numa_action[:self.item_numa_num[item_i]] = 1
        else:
            box_j_numa = self.numa_remain_resources[box_j].reshape(self.max_box_numa, self.len_numa_res)
            item_i_numa = dcp(self.item_numa_res[item_i])
            numa_idxes = np.where((box_j_numa >= item_i_numa).all(axis=-1))[0]
            if len(numa_idxes) <= 0:
                numa_idxes = np.arange(self.max_box_numa)
            # selected_numa = numa_idxes[np.argsort((item_i_numa / (box_j_numa + 1)).mean(axis=-1)[numa_idxes])[:self.item_numa_num[item_i]]]
            item_i_numa[item_i_numa==0] = 1
            selected_numa = numa_idxes[np.argsort(((box_j_numa % item_i_numa) / item_i_numa).mean(axis=-1)[numa_idxes])][:self.item_numa_num[item_i]]
            # selected_numa = np.random.choice(numa_idxes, replace=False, size=self.item_numa_num[item_i])
            if len(selected_numa) < self.item_numa_num[item_i]:
                selected_numa = list(set(list(range(self.max_box_numa))) - set(selected_numa))[:self.item_numa_num[item_i]-len(selected_numa)] + list(selected_numa)
                numa_action[selected_numa] = 1
            else:
                numa_action[selected_numa] = 1
        return numa_action
    
    def step(self, action, numa_action=None):
        if numa_action is not None:
            numa_action = np.array(numa_action)
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[action[0]]).reshape(-1)
            if (self.box_cur_state[action[1], 1] <= 0) and (self.numa_remain_resources[action[1]] < item_i_numa).any():
                numa_action = None
        if (numa_action is None) and (self.len_numa_res > 0):
            numa_action = self.find_numa_placement(action)
        if (numa_action is not None):
            if (len(numa_action) < self.max_box_numa):
                numa_action_t = np.zeros(self.max_box_numa, dtype=int)
                numa_action_t[:len(numa_action)] = numa_action
                numa_action = numa_action_t
            numa_action = np.array(numa_action)
        item_i, box_k = action
        self.cur_placement[item_i, box_k] = 1
        if self.len_numa_res > 0:
            self.cur_items_numa_actions[item_i] = 0
            self.cur_items_numa_actions[item_i, box_k] = numa_action
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
            self.item_cur_numa[item_i] = item_i_numa
        
        if self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(self.resource_types):] -= self.item_cur_state[item_i]
            available = (np.expand_dims(self.box_cur_state[box_k, -len(self.resource_types):], 0).repeat(len(self.item_cur_state), 0) >= self.item_cur_state).all(axis=-1)
            if self.len_numa_res > 0:
                self.numa_remain_resources[box_k] -= self.item_cur_numa[item_i]
                box_numa_res = (np.expand_dims(self.numa_remain_resources[box_k], 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), self.max_box_numa, self.len_numa_res))
                item_numa_res = np.expand_dims(self.item_numa_res, 1).repeat(self.max_box_numa, 1)
                numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num
            else:
                numa_resource_enough = np.ones_like(available, dtype=bool)
        else:
            available = np.ones(len(self.item_cur_state), dtype=bool)
            numa_resource_enough = np.ones_like(available, dtype=bool)
        available[item_i] = False
        
        self.actions_available[:, box_k] = available & (self.item_mutex_box[:, box_k] == 0) & numa_resource_enough
    
    def get_available_actions(self):
        return self.actions_available
    
    def get_item_max_assign_cost_gain(self, placement):
        item_assign_cost = self.item_assign_box_cost.copy()
        item_assign_cost[self.item_mutex_box==1] = 1e20
        return item_assign_cost[placement==1] - item_assign_cost.min(axis=-1)
    
    def get_item_rewards(self, item_i):
        if self.is_filter_unmovable:
            item_assign_costs = self.norm_item_assign_costs[self.item_init_movable]
        else:
            item_assign_costs = self.norm_item_assign_costs
        # print(item_assign_costs.shape, self.init_placement_copy.shape)
        return - (item_assign_costs[item_i] - (item_assign_costs[item_i] * self.init_placement_copy[item_i]).sum() + ((1 - self.init_placement_copy[item_i]) * (1 + self.norm_item_migrations_costs[item_i])))
    
    def _get_reward_batch(self, box_list):
        item_assign_costs = self.item_assign_box_cost[:, box_list] - (self.item_assign_box_cost * self.cur_placement).sum(axis=1).reshape(-1, 1)
        migration_costs = self.item_infos['migrationCost'].values.reshape(-1, 1).repeat(len(box_list), axis=1)
        des_box_used_costs = self.box_cur_state[box_list, 0] * (self.cur_placement[:, box_list].sum(axis=0)<=0)
        item_cur_box = np.where(self.cur_placement==1)[1]
        origin_box_used_costs = (self.box_cur_state[:, 0] * (self.cur_placement.sum(axis=0)==1))[item_cur_box]
        box_used_costs = (des_box_used_costs[None].repeat(len(self.item_cur_state), axis=0) - origin_box_used_costs[:, None].repeat(len(box_list), axis=1))
        reward =  (- item_assign_costs - migration_costs - box_used_costs) # * self.actions_available[:, box_list]
        return reward
    
    def get_cur_state_score(self, is_contain_migration=False, is_normcost=False):
        if self.is_filter_unmovable:
            cur_placement = np.array(self.init_placement).copy()
            cur_placement[self.item_init_movable] = self.cur_placement.copy()
        else:
            cur_placement = self.cur_placement.copy()
        if is_normcost:
            box_used_costs = self.norm_box_used_costs
            item_assign_costs = self.norm_item_assign_costs
            item_migrations_costs = self.norm_item_migrations_costs
        else:
            box_used_costs = self.box_cur_state[:, 0]
            item_assign_costs = self.item_assign_box_cost_origin
            item_migrations_costs = self.item_infos['migrationCost'].values
        used_costs = box_used_costs[cur_placement.sum(axis=0)>0].sum()
        assign_costs = (item_assign_costs * cur_placement).sum()
        if is_contain_migration:
            migration_cost = (item_migrations_costs[(self.init_placement != cur_placement).any(axis=-1)] + 1).sum()
            # if is_normcost:
                # migration_cost = 1000 * (migration_cost / len(self.item_cur_state))
            #print((self.init_placement != cur_placement).any(axis=-1).astype(float).sum())
        else:
            migration_cost = 0
        # print(assign_costs, migration_cost, self.get_cur_state_score_origin())
        return used_costs + assign_costs + migration_cost, (self.init_placement != cur_placement).any(axis=-1).astype(float).sum()
    
    def get_cur_state_score_origin(self):
        if self.is_filter_unmovable:
            cur_placement = np.array(self.init_placement).copy()
            cur_placement[self.item_init_movable] = self.cur_placement.copy()
        else:
            cur_placement = self.cur_placement.copy()

        box_used_costs = self.box_cur_state[:, 0]
        item_assign_costs = self.item_assign_box_cost_origin
        item_migrations_costs = self.item_infos['migrationCost'].values
        used_costs = box_used_costs[cur_placement.sum(axis=0)>0].sum()
        assign_costs = (item_assign_costs * cur_placement).sum()
        migration_cost = (item_migrations_costs[(self.init_placement != cur_placement).any(axis=-1)] + 1).sum()
        return used_costs + assign_costs + migration_cost
