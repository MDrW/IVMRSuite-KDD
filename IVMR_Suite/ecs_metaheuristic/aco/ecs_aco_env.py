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
from ecs_env.ecs_base_env import ECSEnvironment
from copy import deepcopy as dcp
import pdb


class ECSACOEnvironment:
    def __init__(self, data_path, seed=None, is_process_numa=False):
        self.item_infos = pd.read_csv(os.path.join(data_path, 'item_infos.csv'), sep=',', header=0)
        self.box_infos = pd.read_csv(os.path.join(data_path, 'box_infos.csv'), sep=',', header=0)
        
        self.item_full_infos = pd.read_pickle(os.path.join(data_path, 'item_infos.pkl'))
        self.box_full_infos = pd.read_pickle(os.path.join(data_path, 'box_infos.pkl'))
        
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)
        self.resource_types = self.configs['resource_types']
        
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
        
        self.init_placement_origin = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost_origin = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz')).toarray()
        self.item_mutex_box_origin = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz')).toarray()
        
        seed = 0 if seed is None else seed
        self.seed(seed)
        self.reset()
    
    def reset(self):
        # item_cur_state: max_move_count, is_infinite, canMigrate
        self.item_assign_box_cost, self.item_mutex_box = self.item_assign_box_cost_origin.copy(), self.item_mutex_box_origin.copy()
        self.init_placement = np.array(self.init_placement_origin).copy()
        self.item_cur_state = self.item_infos[['count', 'isInfinite', 'canMigrate', 'migrationCost'] + self.resource_types].values
        self.item_cur_state[:, 0] = 1
        self.item_init_movable = self.item_cur_state[:, 2] > 0
        
        self.item_cur_box = np.ones(len(self.item_cur_state), dtype=int) * -1
        self.infinite_item_cur_box = np.zeros((len(self.item_cur_state), len(self.box_infos)), dtype=int)
        self.box_dict = {}
        for i in range(len(self.box_infos)):
            self.box_dict[self.box_infos.loc[i, 'id']] = i
        
        self.item_assign_box_cost -= self.item_assign_box_cost[self.init_placement==1].reshape(-1, 1)
        # pdb.set_trace()
        self.item_assign_box_cost = ((self.item_assign_box_cost / (np.abs(self.item_assign_box_cost).max() + 1e-5))) * 100
        self.item_cur_state[:, 3] = ((self.item_cur_state[:, 3] / (np.abs(self.item_cur_state).max() + 1e-5))) + 1
        # pdb.set_trace()
        self.item_cur_state = self.item_cur_state[self.item_init_movable]
        self.item_cur_box = self.item_cur_box[self.item_init_movable]
        self.item_assign_box_cost = self.item_assign_box_cost[self.item_init_movable]
        self.item_mutex_box = self.item_mutex_box[self.item_init_movable]
        self.init_placement = self.init_placement[self.item_init_movable]
        
        self.item_cur_numa = dcp(self.init_items_numa)
        # self.item_new_numa, self.stored_numa_actions = {}, {}
        self.box_cur_numa = dcp(self.init_boxes_numa)
        self.item_numa_res = dcp(self.init_items_numa_res)
        self.item_numa_num = dcp(self.init_items_numa_num)
        if self.len_numa_res > 0:
            self.item_cur_numa = self.item_cur_numa[self.item_init_movable]
            self.item_numa_res = self.item_numa_res[self.item_init_movable]
            self.item_numa_num = self.item_numa_num[self.item_init_movable] 
        
        i, self.init_idxes_map, self.idxes_to_init_map = 0, {}, {}
        for ri in range(len(self.item_init_movable)):
            if self.item_init_movable[ri]:
                self.init_idxes_map[ri] = i
                self.idxes_to_init_map[i] = ri
                i += 1
        
        self.box_cur_state = self.box_infos[['cost', 'isInfinite'] + self.resource_types].values
        fixed_placement = self.init_placement_origin.copy()
        fixed_placement[((self.item_infos['count'].values <= 0) | (self.item_infos['canMigrate'].values <= 0)).astype(int) <= 0] = 0
        box_fixed_remain_resources = self.box_infos[self.resource_types].values - fixed_placement.T.dot(self.item_infos[self.resource_types].values)
        box_fixed_remain_resources[self.box_infos['isInfinite'].astype(bool).values, :] = (self.item_infos[self.resource_types].values.sum(axis=0) * 2).reshape(-1)
        self.box_cur_state[:, -len(self.resource_types):] = box_fixed_remain_resources
        self.box_cur_state[:, 0] = ((self.box_cur_state[:, 0] / (np.abs(self.box_cur_state[:, 0]).max() + 1e-5)))
        self.box_is_used = np.zeros(len(self.box_cur_state), dtype=bool)
        
        if self.len_numa_res > 0:
            self.numa_remain_resources = self.box_cur_numa - fixed_placement.T.dot(self.init_items_numa)
            self.numa_remain_resources[self.box_infos['isInfinite'].astype(bool).values] = 0
        
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(self.resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources, 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(self.box_cur_numa), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(self.box_cur_numa), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
       
        self.actions_available = ((self.item_mutex_box == 0) & resource_enough & numa_resource_enough).astype(int)
        # pdb.set_trace()
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        self.init_actions_available = self.actions_available.copy()
        
        self.item_count, self.box_count = len(self.item_cur_state), len(self.box_cur_state)
        self.cur_placement = np.zeros((self.item_count, self.box_count), dtype=int)
        
        assign_score = self.item_assign_box_cost.copy()
        assign_score[self.actions_available==0] = np.inf
        # assign_score = -(assign_score.max(axis=-1) - assign_score.min(axis=-1))
        assign_score = assign_score.min(axis=-1) - (self.item_assign_box_cost * self.init_placement).sum(axis=-1)
        available_score = self.actions_available.sum(axis=-1)
        self.item_sorted_idxes = np.lexsort((assign_score, available_score))
        # self.item_sorted_idxes = np.lexsort((available_score, assign_score))
        # pdb.set_trace()
        # gap = int(self.item_count * 0.8)
        # for i in range(int(np.ceil(self.item_count/gap))):
        #     np.random.shuffle(self.item_sorted_idxes[i*gap:(i+1)*gap])
    
    def recovery_to_init_placement(self):
        if (self.cur_placement.sum(axis=-1) <= 0).any():
            self.cur_placement = dcp(self.init_placement)
            self.item_cur_numa = dcp(self.init_items_numa[self.item_init_movable])
    
    def get_item_available_actions(self, item_i):
        return self.actions_available[item_i].astype(bool)
    
    def get_item_rewards(self, item_i):
        return - (self.item_assign_box_cost[item_i] + ((1 - self.init_placement[item_i]) * (1 + self.item_cur_state[item_i, 3])))
    
    def recovery_to_feasible_placement_by_items(self):
        recovery_boxes = (self.box_cur_state[:, -len(self.resource_types):] < 0).any(axis=-1)
        if self.len_numa_res > 0:
            recovery_numa_boxes = (self.numa_remain_resources < 0).any(axis=-1)
            recovery_boxes = (recovery_boxes | recovery_numa_boxes)
        recovery_boxes[self.box_cur_state[:, 1] > 0] = False
        recovery_box_idxes = np.where(recovery_boxes)[0]
        recovery_item_idxes = np.where(self.cur_placement[:, recovery_box_idxes] > 0)[0]
        # pdb.set_trace()
        if len(recovery_item_idxes) <= 0:
            return recovery_item_idxes
        
        release_resources = self.cur_placement[recovery_item_idxes].T.dot(self.item_cur_state[recovery_item_idxes, -len(self.resource_types):])
        self.box_cur_state[:, -len(self.resource_types):] += release_resources
        if self.len_numa_res > 0:
            release_numa_resources = self.cur_placement[recovery_item_idxes].T.dot(self.item_cur_numa[recovery_item_idxes])
            self.numa_remain_resources += release_numa_resources
            self.item_cur_numa[recovery_item_idxes] = np.zeros_like(self.item_cur_numa[recovery_item_idxes], dtype=float)
        
        self.cur_placement[recovery_item_idxes] = 0
        self.item_cur_box[recovery_item_idxes] = -1
        self.item_cur_state[recovery_item_idxes, 0] = 1
        
        resource_enough = (np.expand_dims(self.box_cur_state[:, -len(self.resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources, 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(self.box_cur_numa), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(self.box_cur_numa), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[:, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
        self.actions_available = ((self.item_mutex_box == 0) & resource_enough & numa_resource_enough).astype(int)
        self.actions_available[self.item_cur_state[:, 0]<=0, :] = 0
        self.actions_available[self.item_cur_state[:, 2]<=0, :] = 0
        
        assign_score = self.item_assign_box_cost.copy()
        assign_score[self.actions_available==0] = np.inf
        # assign_score = -(assign_score.max(axis=-1) - assign_score.min(axis=-1))
        assign_score = (assign_score.min(axis=-1) - (self.item_assign_box_cost * self.init_placement).sum(axis=-1))[recovery_item_idxes]
        available_score = self.actions_available.sum(axis=-1)[recovery_item_idxes]
        item_sorted_idxes = recovery_item_idxes[np.lexsort((assign_score, available_score))]
        # if len(item_sorted_idxes) < 150:
        #     pdb.set_trace()
        return item_sorted_idxes
    
    def recovery_to_feasible_placement_by_boxes(self):
        # origin_placement = self.cur_placement.copy()
        recovery_boxes = (self.box_cur_state[:, -len(self.resource_types):] < 0).any(axis=-1)
        if self.len_numa_res > 0:
            recovery_numa_boxes = (self.numa_remain_resources < 0).any(axis=-1)
            recovery_boxes = (recovery_boxes | recovery_numa_boxes)
        recovery_boxes[self.box_cur_state[:, 1] > 0] = False
        recovery_box_idxes = np.where(recovery_boxes)[0]
        
        recovery_flag = np.ones(len(recovery_box_idxes), dtype=bool)
        for bi, box_idx in enumerate(recovery_box_idxes):
            item_idxes = np.where(self.cur_placement[:, box_idx] > 0)[0]
            # pdb.set_trace()
            if len(item_idxes) <= 0:
                recovery_flag[bi] = False
                continue
            
            if self.len_numa_res > 0:
                box_resources = np.concatenate([self.box_cur_state[box_idx, -len(self.resource_types):], self.numa_remain_resources[box_idx]], axis=-1)
                item_resources = np.concatenate([self.item_cur_state[item_idxes, -len(self.resource_types):], self.item_cur_numa[item_idxes]], axis=-1)
            else:
                box_resources = self.box_cur_state[box_idx, -len(self.resource_types):]
                item_resources = self.item_cur_state[item_idxes, -len(self.resource_types):]
            resource_not_enough = box_resources < 0
            assert resource_not_enough.astype(int).sum() > 0
            resource_to_be_enough = (item_resources[:, resource_not_enough] / -box_resources[resource_not_enough])
            # pdb.set_trace()
            item_idxes = item_idxes[resource_to_be_enough.sum(axis=-1) > 0]
            # print(resource_to_be_enough.shape, (resource_to_be_enough.sum(axis=-1) > 0).shape, item_idxes.shape)
            resource_to_be_enough = resource_to_be_enough[(resource_to_be_enough.sum(axis=-1) > 0)][((self.actions_available[item_idxes]==1).any(axis=-1))]
            item_idxes = item_idxes[(self.actions_available[item_idxes]==1).any(axis=-1)]
            # pdb.set_trace()
            if (len(item_idxes) <= 0) or (resource_to_be_enough.sum(axis=0) < 1).any():
                recovery_flag[bi] = False
                continue
            costs = self.item_assign_box_cost[item_idxes].copy()
            cur_costs = (costs * self.cur_placement[item_idxes]).sum(axis=-1)
            costs[self.actions_available[item_idxes] <= 0] = np.inf
            costs_change = costs.min(axis=-1) - cur_costs
            resource_to_be_enough_prob = -resource_to_be_enough.sum(axis=-1)
            item_idxes = item_idxes[np.lexsort((resource_to_be_enough_prob, costs_change))]
            # pdb.set_trace()
            rf = False
            for item_i in item_idxes:
                if self.actions_available[item_i].sum() <= 0:
                    continue
                box_costs = self.item_assign_box_cost[item_i].copy()
                box_costs[self.actions_available[item_i] <= 0] = np.inf
                target_box = np.lexsort((1 - self.init_placement[item_i], box_costs))[0]
                self.step([item_i, target_box])
                if (self.box_cur_state[box_idx, -len(self.resource_types):] >= 0).all() and (self.numa_remain_resources[box_idx] >= 0).all():
                    rf = True
                    break
            # pdb.set_trace()
            recovery_flag[bi] = rf
        # pdb.set_trace()
        # if not recovery_flag:
        #     self.cur_placement = origin_placement
        #     self.cur_placement[:, recovery_box_idxes] = 0
        return recovery_flag.all()
    
    def get_final_reward(self):
        if (self.cur_placement.sum(axis=-1) <= 0).any():
            return -(self.cur_placement.sum(axis=-1) <= 0).astype(int).sum() * 10
            # return 1.
        else:
            assign_cost = (self.item_assign_box_cost * self.cur_placement).sum() - (self.item_assign_box_cost * self.init_placement).sum()
            migrate_cost = (((self.cur_placement - self.init_placement)==1).any(axis=-1).astype(float) * (1 + self.item_cur_state[:, 3])).sum() / self.item_count
            migrate_cost -= (1 + self.item_cur_state[:, 3]).max()
            used_box_cost = self.box_cur_state[self.box_is_used, 0].sum()
            # pdb.set_trace()
            return -(assign_cost + 10 * migrate_cost + used_box_cost)
            # return -assign_cost
    
    def get_final_real_cost(self):
        if (self.cur_placement.sum(axis=-1) <= 0).any():
            return np.inf
        else:
            final_placement = self.get_final_origin_placement()
            assign_cost = (self.item_assign_box_cost_origin * final_placement).sum()
            migrate_cost = (((final_placement - self.init_placement_origin)==1).any(axis=-1).astype(float) * (self.item_infos['migrationCost'].values)).sum()
            # print(((final_placement - self.init_placement_origin)==1).any(axis=-1).astype(float).sum())
            used_box_cost = self.box_cur_state[self.box_is_used, 0].sum()
            return (assign_cost + used_box_cost + migrate_cost)
    
    def get_final_real_cost_dict(self):
        final_placement = self.init_placement_origin.copy()
        final_placement[self.item_init_movable] = self.cur_placement
        assign_cost = (self.item_assign_box_cost_origin * final_placement).sum()
        num_migration = ((final_placement - self.init_placement_origin)==1).any(axis=-1).astype(float).sum()
        migrate_cost = (((final_placement - self.init_placement_origin)==1).any(axis=-1).astype(float) * (self.item_infos['migrationCost'].values)).sum()
        # print(((final_placement - self.init_placement_origin)==1).any(axis=-1).astype(float).sum())
        used_box_cost = self.box_cur_state[self.box_is_used, 0].sum()
        return {'final_score': assign_cost+used_box_cost, 'migrate_cost': migrate_cost, 'num_migration': num_migration}
    
    def get_final_origin_placement(self):
        final_placement = self.init_placement_origin.copy()
        if (self.cur_placement.sum(axis=-1) <= 0).any():
            pass
        else:
            final_placement[self.item_init_movable] = self.cur_placement
        return final_placement
    
    def get_final_placement(self):
        return self.cur_placement.copy()
    
    def seed(self, _seed):
        random.seed(_seed)
        np.random.seed(_seed)
    
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
    
    def get_feasible_numa_actions(self, item_i, box_j):
        box_j_numa = self.numa_remain_resources[box_j].reshape(self.max_box_numa, self.len_numa_res)
        item_i_numa = self.item_numa_res[item_i]
        numa_idxes = np.where((box_j_numa >= item_i_numa).all(axis=-1))[0]
        numa_available = np.zeros(self.max_box_numa, dtype=bool)
        numa_available[numa_idxes] = True
        return numa_available
    
    def get_numa_actions_heuristic(self, item_i, box_j):
        box_j_numa = self.numa_remain_resources[box_j].reshape(self.max_box_numa, self.len_numa_res)
        item_i_numa = self.item_numa_res[item_i]
        item_i_numa[item_i_numa==0] = 1
        return (1 - ((box_j_numa % item_i_numa) / item_i_numa).mean(axis=-1)) * 5
    
    def get_stored_numa_actions(self):
        if self.len_numa_res > 0:
            numa_actions = (self.item_cur_numa.reshape(len(self.item_cur_numa), self.max_box_numa, self.len_numa_res) > 0).any(axis=-1).astype(int)
        else:
            numa_actions = np.zeros((self.item_count, self.max_box_numa), dtype=int)
        items, boxes = np.where(self.cur_placement == 1)
        stored_numa_actions = np.zeros((self.item_count, self.box_count, self.max_box_numa), dtype=int)
        for i in range(len(items)):
            stored_numa_actions[items[i], boxes[i]] = numa_actions[i]
        return stored_numa_actions
    
    def step(self, action, numa_action=None):
        if numa_action is not None:
            numa_action = np.array(numa_action)
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[action[0]]).reshape(-1)
            if (self.box_cur_state[action[1], 1] <= 0) and (self.numa_remain_resources[action[1]] < item_i_numa).any():
                numa_action = None
        # numa_action = self.stored_numa_actions.get(tuple(action)) if numa_action is None else numa_action
        if (numa_action is None) and (self.len_numa_res > 0):
            numa_action = self.find_numa_placement(action)
        if (numa_action is not None):
            if (len(numa_action) < self.max_box_numa):
                numa_action_t = np.zeros(self.max_box_numa, dtype=int)
                numa_action_t[:len(numa_action)] = numa_action
                numa_action = numa_action_t
            numa_action = np.array(numa_action)
        
        self._update_state(action, numa_action)
    
    def _update_box_action_available(self, box_list):
        resource_enough = (np.expand_dims(self.box_cur_state[box_list, -len(self.resource_types):], 0).repeat(len(self.item_cur_state), 0) >= np.expand_dims(self.item_cur_state[:, -len(self.resource_types):], 1)).all(axis=-1)
        resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        
        if self.len_numa_res > 0:
            box_numa_res = (np.expand_dims(self.numa_remain_resources[box_list], 0).repeat(len(self.item_cur_state), 0).reshape(len(self.item_cur_state), len(box_list), self.max_box_numa, self.len_numa_res))
            item_numa_res = np.expand_dims((np.expand_dims(self.item_numa_res, 1).repeat(len(box_list), 1)), 2).repeat(self.max_box_numa, 2)
            numa_resource_enough = (((box_numa_res >= item_numa_res).all(axis=-1)).astype(int).sum(axis=-1)) >= self.item_numa_num.reshape(-1, 1)
            numa_resource_enough[:, self.box_cur_state[box_list, 1]>=1] = True
        else:
            numa_resource_enough = np.ones_like(resource_enough, dtype=bool)
        
        self.actions_available[:, box_list] = ((self.item_mutex_box[:, box_list] == 0) & resource_enough & numa_resource_enough).astype(int)
    
    def _update_state(self, action, numa_action):
        if numa_action is not None:
            numa_action = np.array(numa_action)
        
        item_i, box_k = action
        # update placement matrix
        self.cur_placement[item_i, box_k] = 1
        box_j = self.item_cur_box[item_i]
        if box_j >= 0:
            self.cur_placement[item_i, box_j] = 0
        self.item_cur_box[item_i] = box_k
        if self.item_cur_state[item_i, 1] > 0:
            self.infinite_item_cur_box[item_i, box_k] += 1
        # update box resource
        if numa_action is not None:
            item_i_numa = (numa_action.reshape(-1, 1).repeat(self.len_numa_res, 1) * self.item_numa_res[item_i]).reshape(-1)
        else:
            item_i_numa = None
        
        if self.box_cur_state[box_k, 1] <= 0:
            self.box_cur_state[box_k, -len(self.resource_types):] -= self.item_cur_state[item_i, -len(self.resource_types):]
            if numa_action is not None:
                self.numa_remain_resources[box_k] -= item_i_numa
        self.box_is_used[box_k] = True
        if (self.box_cur_state[box_j, 1] <= 0) and (box_j >= 0):
            self.box_cur_state[box_j, -len(self.resource_types):] += self.item_cur_state[item_i, -len(self.resource_types):]
            if numa_action is not None:
                self.numa_remain_resources[box_j] += self.item_cur_numa[item_i]
            self.box_is_used[box_j] = self.cur_placement[:, box_j].sum() > 0
        if (self.len_numa_res > 0) and (numa_action is not None):
            # self.item_new_numa[(item_i, box_k)] = item_i_numa
            # self.stored_numa_actions[(item_i, box_k)] = numa_action
            self.item_cur_numa[item_i] = item_i_numa
        
        # update item canMigrate
        if (self.item_cur_state[item_i, 1] <= 0):
            self.item_cur_state[item_i, 0] -= 1
        
        update_boxes = [box_j, box_k] if box_j >=0 else [box_k]
        self._update_box_action_available(update_boxes)     


if __name__ == '__main__':
    pass
