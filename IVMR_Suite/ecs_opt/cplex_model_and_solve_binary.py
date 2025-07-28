import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy import sparse
import json
import time
from docplex.mp.model import Model
from docplex.mp.model_reader import ModelReader
from copy import deepcopy as dcp
import pdb


class ECSCplexOptModel:
    def __init__(self, data_path, model_save_path=None, logger_path=None, 
                 cplex_params={}, add_migration_cost=0., is_remove_unmovable=False):
        with open(os.path.join(data_path, 'config.json'), 'r') as f:
            self.configs = json.load(f)
        
        self.item_infos = pd.read_pickle(os.path.join(data_path, 'item_infos.pkl'))
        self.box_infos = pd.read_pickle(os.path.join(data_path, 'box_infos.pkl'))
        # self.item_count, self.box_count = len(self.item_infos), len(self.box_infos)
        
        self.init_placement = sparse.load_npz(os.path.join(data_path, 'init_placement.npz')).toarray()
        self.item_assign_box_cost = sparse.load_npz(os.path.join(data_path, 'item_assign_box_cost.npz'))
        self.item_mutex_box = sparse.load_npz(os.path.join(data_path, 'item_mutex_box.npz'))
        self.item_mix_item_cost = sparse.load_npz(os.path.join(data_path, 'item_mix_item_cost.npz'))
        self.item_mutex_item = sparse.load_npz(os.path.join(data_path, 'item_mutex_item.npz'))
        
        self.item_infos['migrationCost'] += add_migration_cost
        
        if logger_path is not None:
            self.logger = open(logger_path, 'w')
        else:
            self.logger = None
        
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.cplex_params = cplex_params
        self.solution = None
        
        if is_remove_unmovable:
            self._remove_unmovable_items()
        else:
            self.item_assign_box_cost = self.item_assign_box_cost.tocoo()
            self.item_mutex_box = self.item_mutex_box.tocoo()
            self.item_mix_item_cost = self.item_mix_item_cost.tocoo()
            self.item_mutex_item = self.item_mutex_item.tocoo()
        self.item_count, self.box_count = len(self.item_infos), len(self.box_infos)
        
        if self.cplex_params.get('timelimit') is None:
            if self.item_count < 5000:
                self.cplex_params['timelimit'] = 7200
            elif self.item_count < 10000:
                self.cplex_params['timelimit'] = 10800
            elif self.item_count < 20000:
                self.cplex_params['timelimit'] = 21600
            else:
                self.cplex_params['timelimit'] = 36000
        
        self.item_item_relation_idxes = set(self.item_mix_item_cost.row).union(self.item_mix_item_cost.col).union(self.item_mutex_item.row).union(self.item_mutex_item.col)
        self.item_item_relation_idxes = sorted(self.item_item_relation_idxes)
        self.item_item_num = len(self.item_item_relation_idxes)
        # self.item_item_dict = dict(zip(self.item_item_relation_idxes, range(self.item_item_num)))
        
        self.penalty_coef = self.item_infos['migrationCost'].max() + self.box_infos['cost'].max()
        if len(self.item_assign_box_cost.data) > 0:
            self.penalty_coef += self.item_assign_box_cost.data.max() - min(self.item_assign_box_cost.data.min(), 0)
        if len(self.item_mix_item_cost.data) > 0:
            self.penalty_coef += self.item_mix_item_cost.data.max() - min(self.item_mix_item_cost.data.min(), 0)
        if (self.penalty_coef <= 0) or (self.configs['fixConflictOnly'] > 0):
            self.penalty_coef = 1.
        self.penalty_coef *= 100
        # self.penalty_coef = 1e33
        
        self._write_info(f"Start Opt Solving {data_path}...")
        # pdb.set_trace()
    
    def _remove_unmovable_items(self):
        self.boxes_id_to_index = {}
        for i in range(len(self.box_infos)):
            self.boxes_id_to_index[self.box_infos.iloc[i]['id']] = i
        self.movable_items_flag = (self.item_infos['canMigrate'] > 0).values.copy()
        
        self.resource_types, self.numa_resource_types = self.configs['resource_types'], self.configs['compound_resource_types']['numa']
        for i in range(len(self.movable_items_flag)):
            if not self.movable_items_flag[i]:
                box_idx = self.boxes_id_to_index[self.item_infos.loc[i, 'inBox']]
                if self.box_infos.loc[box_idx, 'isInfinite'] <= 0:
                    self.box_infos.loc[box_idx, self.resource_types] -= self.item_infos.loc[i, self.resource_types]
                    # self.box_infos.loc[box_idx, self.resource_types][self.box_infos.loc[box_idx, self.resource_types] < 0] = 0
                    
                    item_numa = self.item_infos.loc[i, 'numa']
                    if item_numa[1] > 0:
                        for numa_i in item_numa[0]:
                            for ri in range(len(self.numa_resource_types)):
                                self.box_infos.loc[box_idx, 'numa'][int(numa_i)][ri] -= item_numa[2+ri]
                                # if self.box_infos.loc[box_idx, 'numa'][int(numa_i)][ri] < 0:
                                #     self.box_infos.loc[box_idx, 'numa'][int(numa_i)][ri] = 0
        
        self.item_infos = pd.DataFrame(self.item_infos[self.movable_items_flag]).reset_index(drop=True)
        
        unmovable_placement = self.init_placement[~self.movable_items_flag]
        self.init_assign_cost = (self.item_assign_box_cost[~self.movable_items_flag].toarray() * unmovable_placement).sum()
        self.init_mix_cost = (self.item_mix_item_cost[~self.movable_items_flag][:, ~self.movable_items_flag].toarray() * unmovable_placement[:, np.where(unmovable_placement==1)[1]]).sum()
        self.init_box_used = (unmovable_placement.sum(axis=0) > 0)
        
        additional_item_mutex_box = np.matmul(self.item_mutex_item[:, ~self.movable_items_flag].toarray(), self.init_placement[~self.movable_items_flag])
        additional_item_assign_box = np.matmul(self.item_mix_item_cost[:, ~self.movable_items_flag].toarray(), self.init_placement[~self.movable_items_flag])
        
        self.init_placement = self.init_placement[self.movable_items_flag]
        self.item_assign_box_cost = sparse.coo_matrix((self.item_assign_box_cost.toarray() + additional_item_assign_box)[self.movable_items_flag])
        self.item_mutex_box = sparse.coo_matrix(((self.item_mutex_box.toarray() + additional_item_mutex_box) > 0).astype(int)[self.movable_items_flag])
        # self.item_assign_box_cost = (self.item_assign_box_cost[self.movable_items_flag]).tocoo()
        # self.item_mutex_box = (self.item_mutex_box[self.movable_items_flag]).tocoo()
        self.item_mix_item_cost = self.item_mix_item_cost[self.movable_items_flag][:, self.movable_items_flag].tocoo()
        self.item_mutex_item = self.item_mutex_item[self.movable_items_flag][:, self.movable_items_flag].tocoo()
    
    def _write_info(self, info):
        if self.logger is not None:
            self.logger.write((info+'\n'))
            self.logger.flush()
        else:
            print(info)
    
    def _define_vars(self):
        self.vars = self.ecs_model.binary_var_matrix(self.item_count, self.box_count, name='x')
        if len(self.item_mix_item_cost.row) > 0:
            self.iic_vars = self.ecs_model.binary_var_dict(
                ((i, j, k) for i, j in zip(self.item_mix_item_cost.row, self.item_mix_item_cost.col) for k in range(self.box_count)),
                name='iic'
            )
        if len(self.item_mutex_item.row) > 0:
            self.iim_vars = self.ecs_model.binary_var_dict(
                ((i, j, k) for i, j in zip(self.item_mutex_item.row, self.item_mutex_item.col) for k in range(self.box_count)),
                name='iim'
            )
    
    def _set_objective(self):
        # item migration cost
        item_migrate_index = list(self.item_infos[self.item_infos['canMigrate'].astype(int) == 1].index)
        if len(item_migrate_index) > 0:
            item_migration_costs = self.ecs_model.sum([
                (float(self.item_infos.loc[i, 'migrationCost'])) * \
                (self.init_placement[i, np.where(self.init_placement[i] == 1)[0][0]] - self.vars[(i, np.where(self.init_placement[i] == 1)[0][0])])
                for i in item_migrate_index
            ])
        else:
            item_migration_costs = 0.
        
        # box use cost
        box_used_costs = (self.init_box_used.astype(int) * self.box_infos['cost'].astype(float)).sum()
        box_cost_nozero_index = list(self.box_infos[(self.box_infos['cost'].astype(float) != 0.) & (~self.init_box_used)].index)
        if len(box_cost_nozero_index) > 0:
            self.box_used_vars = self.ecs_model.binary_var_list(len(box_cost_nozero_index), lb=0, ub=1, name='bu')
            box_used_costs += self.ecs_model.sum([
                float(self.box_infos.iloc[bj]['cost']) * self.box_used_vars[bi]
                for bi, bj in enumerate(box_cost_nozero_index)
            ])
            box_used_constrs = []
            for j in range(len(box_cost_nozero_index)):
                bu = (self.ecs_model.sum([self.vars[(i,j)] for i in range(self.item_count)]))
                box_used_constrs.append(self.item_count*self.box_used_vars[j] >= bu)
                box_used_constrs.append(self.box_used_vars[j] <= bu)
            self.ecs_model.add_constraints(box_used_constrs)
        
        if self.configs['fixConflictOnly'] > 0:
            self.objective = item_migration_costs + box_used_costs
        else:
            # item assign box cost
            assign_costs = self.init_assign_cost
            if len(self.item_assign_box_cost.row) > 0:
                assign_costs += self.ecs_model.sum([
                    v * self.vars[(i,j)]
                    for i, j, v in zip(self.item_assign_box_cost.row, self.item_assign_box_cost.col, self.item_assign_box_cost.data)
                ])
            
            # item mix item cost
            mix_costs = self.init_mix_cost
            if (len(self.item_mix_item_cost.row) > 0):
                rows, cols, vals = self.item_mix_item_cost.row, self.item_mix_item_cost.col, self.item_mix_item_cost.data
                mix_costs += self.ecs_model.sum([
                    float(v) * self.ecs_model.sum([self.iic_vars[(i, j, k)] for k in range(self.box_count)]) for i, j, v in zip(rows, cols, vals)
                ])
            
            # objective
            self.objective = box_used_costs + assign_costs + item_migration_costs + mix_costs
        # self.ecs_model.set_objective(sense='min', expr=self.objective)
        
    def _set_item_mutex_box(self):
        self.item_mutex_box_cons, self.item_mutex_box_vars = [], []
        for item_i, box_j in zip(self.item_mutex_box.row, self.item_mutex_box.col):
            if self.init_placement[item_i, box_j] > 0:
                ibm = self.ecs_model.continuous_var(lb=0, ub=self.init_placement[item_i, box_j], name=f"ibm_{item_i}_{box_j}")
                self.item_mutex_box_cons.append(self.vars[(item_i, box_j)] <= ibm)
                self.objective += self.penalty_coef * ibm
                self.item_mutex_box_vars.append(ibm)
            else:
                self.item_mutex_box_cons.append(self.vars[(item_i, box_j)] <= 0)
        
        self.ecs_model.add_constraints(self.item_mutex_box_cons)
    
    def _set_item_can_migrate(self):
        self.item_can_migrate_cons = [
            self.vars[(item_i, box_j)] == self.init_placement[item_i, box_j]
            for item_i in np.array(range(self.item_count))[self.item_infos['canMigrate'].astype(int).values<1]
            for box_j in range(self.box_count)
        ]
        self.ecs_model.add_constraints(self.item_can_migrate_cons)
    
    def _set_item_placement_limitation(self):
        self.item_placement_cons = [
            self.ecs_model.sum([
                self.vars[(item_i, box_j)] for box_j in range(self.box_count)
            ]) == self.init_placement[item_i].sum()
            for item_i in np.array(range(self.item_count))[self.item_infos['canMigrate'].astype(int).values>=1]
        ]
        self.ecs_model.add_constraints(self.item_placement_cons)
    
    def _set_item_mutex_item(self):
        self.item_mutex_item_cons, self.item_mutex_item_vars = [], []
        for i, j in zip(self.item_mutex_item.row, self.item_mutex_item.col):
            for k in range(self.box_count):
                if i == j:
                    continue
                if (self.init_placement[i, k] > 0) and (self.init_placement[j, k] > 0):
                    iim = self.ecs_model.continuous_var(lb=0, ub=min(self.init_placement[i, k], self.init_placement[j, k]), name=f"iim_{i}_{j}_{k}")
                    self.item_mutex_item_cons.append(self.iim_vars[(i, j, k)] <= iim)
                    self.objective += self.penalty_coef * iim
                    self.item_mutex_item_vars.append(iim)
                else:
                    self.item_mutex_item_cons.append(self.iim_vars[(i, j, k)] <= 0)
                self.item_mutex_item_cons.append(self.iim_vars[(i, j, k)] <= self.vars[(i, k)])
                self.item_mutex_item_cons.append(self.iim_vars[(i, j, k)] <= self.vars[(j, k)])
                self.item_mutex_item_cons.append(self.iim_vars[(i, j, k)] >= (self.vars[(i, k)] + self.vars[(j, k)] - 1))
        self.ecs_model.add_constraints(self.item_mutex_item_cons)
    
    def _set_item_item_restriction(self):
        if len(self.item_mix_item_cost.row) > 0:
            self.item_item_restriction = []
            for i, j in zip(self.item_mix_item_cost.row, self.item_mix_item_cost.col):
                for k in range(self.box_count):
                    self.item_item_restriction.append(self.iic_vars[(i, j, k)] <= self.vars[(i, k)])
                    self.item_item_restriction.append(self.iic_vars[(i, j, k)] <= self.vars[(j, k)])
                    self.item_item_restriction.append(self.iic_vars[(i, j, k)] >= (self.vars[(i, k)] + self.vars[(j, k)] - 1))
            self.ecs_model.add_constraints(self.item_item_restriction)
    
    def _set_box_simple_resources_satisfaction(self):
        self.box_resources_sat_cons, self.box_resources_vio_cons, self.box_resources_sat_vars = [], [], []
        for box_j in np.array(range(self.box_count))[self.box_infos['isInfinite'].astype(int).values==0]:
            for resource_type in self.configs["resource_types"]:
                init_res = (self.init_placement[:, box_j] * self.item_infos[resource_type].values.astype(float)).sum()
                if init_res > float(self.box_infos.loc[box_j, resource_type]):
                    gap = init_res - float(self.box_infos.loc[box_j, resource_type])
                    brv = self.ecs_model.continuous_var(lb=0, ub=gap, name=f'br_{box_j}_{resource_type}')
                    self.box_resources_sat_cons.append(
                        self.ecs_model.sum([
                            self.vars[(item_i, box_j)] * float(self.item_infos.loc[item_i, resource_type])
                            for item_i in range(self.item_count)
                        ]) <= float(self.box_infos.loc[box_j, resource_type]) + brv
                    )
                    self.objective += brv / gap * self.penalty_coef
                    self.box_resources_sat_vars.append(brv)
                    
                    brv_y = self.ecs_model.binary_var(name=f"br_{box_j}_{resource_type}_y")
                    unplaced_items = np.where(self.init_placement[:, box_j] == 0)[0]
                    self.box_resources_vio_cons.extend([
                        brv <= gap * brv_y,
                        brv >= 0.01 * brv_y,
                        self.ecs_model.sum([self.vars[(item_i, box_j)] for item_i in unplaced_items]) <= len(unplaced_items) * (1 - brv_y)
                    ])
                else:
                    self.box_resources_sat_cons.append(
                        self.ecs_model.sum([
                            self.vars[(item_i, box_j)] * float(self.item_infos.loc[item_i, resource_type])
                            for item_i in range(self.item_count)
                        ]) <= float(self.box_infos.loc[box_j, resource_type])
                    )
        
        self.ecs_model.add_constraints(self.box_resources_sat_cons)
        self.ecs_model.add_constraints(self.box_resources_vio_cons)
    
    def _set_box_numa_resource_satisfaction(self):
        # only for binary variables
        self.box_numa_resources_sat_cons, self.box_numa_sat_vars = [], []
        for box_j in np.array(range(self.box_count))[self.box_infos['isInfinite'].astype(int).values==0]:
            b_numa = self.box_infos.iloc[box_j]['numa']
            numa_resource_types = self.configs['compound_resource_types']['numa']
            l_numa_res = len(numa_resource_types)
            
            numa_constrs = [0 for _ in range(len(b_numa) * l_numa_res)]
            init_numa_res = np.zeros((len(b_numa), l_numa_res))
            consistency_constrs = []
            n = 0
            for item_i in range(len(self.item_infos)):
                i_numa = self.item_infos.iloc[item_i]['numa']
                if len(i_numa[0]) < i_numa[1]:
                    i_numa[0] = list(range(i_numa[1]))
                elif len(i_numa[0]) > i_numa[1]:
                    i_numa[0] = i_numa[0][:i_numa[1]]
                if (i_numa[1] > len(b_numa)):
                    if self.init_placement[item_i, box_j] > 0:
                        raise ValueError(f"Item {item_i} has more numa nodes than box {box_j}!")
                    else:
                        consistency_constrs.append(self.vars[(item_i, box_j)] == 0)
                elif (len(b_numa) <= 0) or (i_numa[1] <= 0):
                    continue
                elif self.item_infos.iloc[item_i]['canMigrate'] == 0:
                    if self.item_infos.iloc[item_i]['inBox'] == self.box_infos.iloc[box_j]['id']:
                        for idx in i_numa[0]:
                            for ri in range(l_numa_res):
                                numa_constrs[int(idx)*l_numa_res+ri] += self.init_placement[item_i, box_j] * i_numa[2+ri]
                                init_numa_res[int(idx), ri] += self.init_placement[item_i, box_j] * i_numa[2+ri]
                else:
                    # numa_vars = self.ecs_model.binary_var_matrix(i_numa[1], len(b_numa), name=f'numa_{item_i}_{box_j}')
                    numa_vars = self.ecs_model.binary_var_list(len(b_numa), lb=0, ub=1, name=f'numa_{item_i}_{box_j}')
                    for bi in range(len(b_numa)):
                        # consistency_constrs.append(self.ecs_model.sum([numa_vars[(ii, bi)] for ii in range(i_numa[1])]) <= self.vars[(item_i, box_j)])
                        consistency_constrs.append(numa_vars[bi] <= self.vars[(item_i, box_j)])
                        for ri in range(l_numa_res):
                            # numa_constrs[bi*l_numa_res+ri] += self.ecs_model.sum([i_numa[2+ri] * numa_vars[(ii, bi)] for ii in range(i_numa[1])])
                            numa_constrs[bi*l_numa_res+ri] += i_numa[2+ri] * numa_vars[bi]
                            n += 1
                            if (self.init_placement[item_i, box_j] > 0) and (str(bi) in i_numa[0]):
                                init_numa_res[bi, ri] += self.init_placement[item_i, box_j] * i_numa[2+ri]
                    
                    if self.item_infos.iloc[item_i]['inBox'] == self.box_infos.iloc[box_j]['id']:
                        for ii, idx in enumerate(i_numa[0]):
                            consistency_constrs.append(numa_vars[int(idx)] == self.vars[(item_i, box_j)])

                    consistency_constrs.append(self.ecs_model.sum([numa_vars[bi] for bi in range(len(b_numa))]) == i_numa[1] * self.vars[(item_i, box_j)])
            # if self.init_placement[:, box_j].sum() > 0:
            #     pdb.set_trace()
            numa_constrs_list = []
            for bi in range(len(b_numa)):
                for ri in range(l_numa_res):
                    if (type(numa_constrs[bi*l_numa_res+ri]) is not int) and (type(numa_constrs[bi*l_numa_res+ri]) is not float):
                        if init_numa_res[bi, ri] > b_numa[bi][ri]:
                            gap = init_numa_res[bi, ri] - b_numa[bi][ri]
                            nr = self.ecs_model.continuous_var(lb=0, ub=gap, name=f'nr_{box_j}_{bi}_{ri}')
                            numa_constrs_list.append(numa_constrs[bi*l_numa_res+ri] <= b_numa[bi][ri] + nr)
                            self.objective += nr / gap * self.penalty_coef
                            self.box_numa_sat_vars.append(nr)
                            
                            nr_y = self.ecs_model.binary_var(name=f"nr_{box_j}_{bi}_{ri}_y")
                            unplaced_items = np.where(self.init_placement[:, box_j] == 0)[0]
                            numa_constrs_list.extend([
                                nr <= gap * nr_y,
                                nr >= 0.01 * nr_y,
                                self.ecs_model.sum([self.vars[(item_i, box_j)] for item_i in unplaced_items]) <= len(unplaced_items) * (1 - nr_y)
                            ])
                        else:
                            numa_constrs_list.append(numa_constrs[bi*l_numa_res+ri] <= b_numa[bi][ri])
            # pdb.set_trace()
            if n > 0:
                consistency_constrs.extend(numa_constrs_list)
            
            self.box_numa_resources_sat_cons.extend(consistency_constrs)
        
        self.ecs_model.add_constraints(self.box_numa_resources_sat_cons)
    
    def _set_max_migration(self):
        item_migrate_index = list(self.item_infos[self.item_infos['canMigrate'].astype(int) == 1].index)
        if len(item_migrate_index) > 0:
            self.max_migration_cons = self.ecs_model.sum([
                (self.init_placement[i, np.where(self.init_placement[i] == 1)[0][0]] - self.vars[(i, np.where(self.init_placement[i] == 1)[0][0])])
                for i in item_migrate_index
            ]) <= self.configs['maxMigration']
        else:
            self.max_migration_cons = 0 <= self.configs['maxMigration']
        self.ecs_model.add_constraint(self.max_migration_cons)
    
    def _set_constraints(self):
        st = time.time()
        # Item Can Migrate
        self._set_item_can_migrate()
        st1 = time.time()
        self._write_info(f"   * Constraints: Item Can Migrate are Defined! Spend Time: {st1-st}s.")
        
        # Item Placement Limitation
        self._set_item_placement_limitation()
        st2 = time.time()
        self._write_info(f"   * Constraints: Item Placement Limitation are Defined! Spend Time: {st2-st1}s.")
        
        # Max Migration Limitation
        self._set_max_migration()
        st3 = time.time()
        self._write_info(f"   * Constraints: Max Migration Count Limitation are Defined! Spend Time: {st3-st2}s.")
        
        # Item Mutex Box Constraints
        self._set_item_mutex_box()
        st4 = time.time()
        self._write_info(f"   * Constraints: Item Mutex Box are Defined! Spend Time: {st4-st3}s.")
        
        # Item Mutex Item Constraints
        self._set_item_mutex_item()
        st5 = time.time()
        self._write_info(f"   * Constraints: Item Mutex Item are Defined! Spend Time: {st5-st4}s.")
        
        # Item Item Restriction
        self._set_item_item_restriction()
        st6 = time.time()
        self._write_info(f"   * Constraints: Item Item Restriction are Defined! Spend Time: {st6-st5}s.")
        
        # Box Simple Resources Satisfaction
        self._set_box_simple_resources_satisfaction()
        st7 = time.time()
        self._write_info(f"   * Constraints: Box Simple Resources Sat are Defined! Spend Time: {st7-st6}s.")
        
        # Compound Resources (Numa) Satisfaction
        self._set_box_numa_resource_satisfaction()
        st8 = time.time()
        self._write_info(f"   * Constraints: Box Compound Resources Numa Sat are Defined! Spend Time: {st8-st7}s.")
    
    def _show_constraints_violation(self):
        self._write_info("Initial Placement Constraints Violation:")
        self._write_info(f"   * Item Mutex Box: {len(self.item_mutex_box_vars)}")
        self._write_info(f"   * Item Mutex Item: {len(self.item_mutex_item_vars)}")
        self._write_info(f"   * Box Simple Resources: {len(self.box_resources_sat_vars)}")
        self._write_info(f"   * Box Compound Resources: {len(self.box_numa_sat_vars)}")
    
    def model(self):
        st1 = time.time()
        if (self.model_save_path is not None) and (str(self.model_save_path).endswith(('.lp', '.lp.gz', '.mps', '.mps.gz'))) and\
           (os.path.exists(self.model_save_path) or os.path.exists(self.model_save_path+'.gz')):
            if os.path.exists(self.model_save_path+'.gz'):
                self.model_save_path = self.model_save_path + '.gz'
            self._write_info(f"Start Reading Model from {self.model_save_path}...")
            self.ecs_model = ModelReader.read(self.model_save_path)
            self._write_info(f"Read Model from {self.model_save_path}.")
        else:
            self._write_info(f"Start Constructing Model {self.data_path}...")
            self.ecs_model = Model(self.data_path)
            
            # define item-box placement variables
            self._define_vars()
            st2 = time.time()
            self._write_info(f"Variables are Defined! Elapsed Time: {st2-st1}s.")
            
            # define objective
            self._set_objective()
            st3 = time.time()
            self._write_info(f"Objective is Defined! Elapsed Time: {st3-st2}s.\nStart Defining Constraints!")
            
            # define constraints
            self._set_constraints()
            self._show_constraints_violation()
            st4 = time.time()
            self._write_info(f"Constraints are Defined! Elapsed Time: {st4-st3}s.")
            
            # set objective
            self.ecs_model.set_objective(sense='min', expr=self.objective)
            
            # write model to mps/lp file
            if self.model_save_path is not None:
                self.ecs_model.export(self.model_save_path)
                # os.system(f"gzip {model_save_path}")
                self._write_info(f"Write Model to {self.model_save_path}! Total Elapsed Time: {time.time()-st4}s.")
        
        self._write_info(f"Model {self.data_path} has been Constructed! Total Elapsed Time: {time.time()-st1}s.")
        # pdb.set_trace()
        return self.ecs_model
    
    def solve(self, sol_save_path=None):
        self._write_info(f"Start Solving {self.data_path}...")
        st = time.time()
        # self.ecs_model.print_information()
        info = f'Model: {self.ecs_model.name}\n' + self.ecs_model.get_statistics().to_string()
        self._write_info(info)
        log_output = True if self.logger is None else self.logger
        self.solution = self.ecs_model.solve(log_output=log_output, cplex_parameters=self.cplex_params)
        st1 = time.time()
        info = f"Model {self.data_path} is Solved by Cplex! Elapsed Time: {st1-st}s."
        self._write_info(info)
        if self.solution:
            self.ecs_model.name = self.ecs_model.name.split('/')[-1]
            info = f"The Optimal Objective: {self.solution.objective_value}! \nSolve Details:\n {self.solution.solve_details}"
            self._write_info(info)
            if sol_save_path is not None and os.path.exists(sol_save_path):
                self.solution.export_as_mst(sol_save_path)
        else:
            info = f"Model {self.data_path} has no solutions! Solve Status: {self.ecs_model.get_solve_status()}!"
            self._write_info(info)
        # pdb.set_trace()
        return self.solution


if __name__ == '__main__':
    data_id = '0'
    data_path = f'../data/ecs_data/{data_id}'
    model_save_path = f'./{data_id}'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    logger_path = None
    cplex_params = {'timelimit': 1000, 'threads': 32, 'mip.tolerances.mipgap': 0.02}
    add_migration_cost = 1.
    st = time.time()
    ecs_opt_model = ECSCplexOptModel(
        data_path=data_path, 
        model_save_path=model_save_path, 
        logger_path=logger_path,
        cplex_params=cplex_params,
        add_migration_cost=add_migration_cost,
        is_remove_unmovable=True,
    )
    
    sol_save_path = f'./{data_id}.mst'
    os.makedirs(os.path.dirname(sol_save_path), exist_ok=True)
    if not os.path.exists(sol_save_path):
        ecs_opt_model.model()
        ecs_opt_model.solve(sol_save_path=os.path.dirname(sol_save_path))
        sol_save_path = None
    end = time.time()
    print("Total Elapsed Time:", end-st)

