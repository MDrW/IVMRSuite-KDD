import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import random
from argparse import ArgumentParser
import time
from ecs_env.ecs_base_env import ECSEnvironment
from ecs_metaheuristic.ga.ecs_ga_env import ECSGAEnvironment
from multiprocessing import Pool
from scipy import sparse
from copy import deepcopy as dcp
import json
import pdb


def _random_move_to_new_placement(idx: int, seed: int, args):
    ecs_env = ECSEnvironment(data_path=args.data_path, is_limited_count=True, is_filter_unmovable=True, is_process_numa=True)
    random.seed(seed)
    np.random.seed(seed)
    # ecs_env.reset()
    done = idx == 0
    while (not done) and (random.uniform(0, 1) < 0.9):
        available = ecs_env.get_available_actions()
        if available.sum() <= 0:
            break
        item_max_gain = ecs_env.item_assign_box_cost.copy()
        item_max_gain[ecs_env.actions_available < 0] = np.inf
        item_max_gain = ((ecs_env.item_assign_box_cost * ecs_env.cur_placement).sum(axis=-1) - item_max_gain.min(axis=-1))[available.sum(axis=1) > 0]
        prob = np.power((item_max_gain - item_max_gain.min() + 1), args.init_mu)
        prob = prob / prob.sum()
        item = np.random.choice(np.where(available.sum(axis=1)>0)[0], size=1, p=prob)[0]
        prob = -(ecs_env.item_assign_box_cost[item])[available[item] > 0]
        prob = np.power((prob - prob.min() + 1), args.init_mu)
        prob = prob / prob.sum()
        box = np.random.choice(np.where(available[item]>0)[0], size=1, p=prob)[0]
        action = [item, box]
        _, _, done, _ = ecs_env.step(action)
    cur_placement = ecs_env.cur_placement.copy()
    items_cur_numa = ecs_env.get_items_cur_numa()
    ecs_env = ECSGAEnvironment(data_path=args.data_path, is_filter_unmovable=True, is_process_numa=True)
    ecs_env.cur_placement = cur_placement.copy()
    score, num_moves = ecs_env.get_cur_state_score(is_contain_migration=args.is_contain_migration, is_normcost=args.is_normcost)
    return cur_placement, items_cur_numa, score, num_moves


def _mutate(placement, numa_actions, seed, args):
    placement, numa_actions = dcp(placement), dcp(numa_actions)
    random.seed(seed)
    np.random.seed(seed)
    ecs_env = ECSGAEnvironment(data_path=args.data_path, is_filter_unmovable=True, is_process_numa=True)
    item_max_gain = ecs_env.get_item_max_assign_cost_gain(placement)
    # additional_val = abs(item_max_gain.max()) + 1
    additional_val = 1
    prob = np.power((item_max_gain - item_max_gain.min() + additional_val), args.mutation_mu)
    prob = prob / prob.sum()
    item_idxes = np.random.choice(range(len(placement)), size=int(len(placement)*args.mutation_prop), replace=False, p=prob)
    placement[item_idxes] = 0
    numa_actions[item_idxes] = 0
        
    return _find_feasible_placement(ecs_env, placement, numa_actions, item_idxes, args)


def _crossover(parent_a, parent_b, numa_actions_a, numa_actions_b, seed, args):
    parent_a, numa_actions_a = dcp(parent_a), dcp(numa_actions_a)
    random.seed(seed)
    np.random.seed(seed)
    ecs_env = ECSGAEnvironment(data_path=args.data_path, is_filter_unmovable=True, is_process_numa=True)
    # item_cost_gain = (ecs_env.item_assign_box_cost * parent_b).sum(axis=-1) - (ecs_env.item_assign_box_cost * ecs_env.cur_placement).sum(axis=-1)
    # box_cost_gain = (parent_b * item_cost_gain.reshape(-1, 1)).sum(axis=0)
    # prob = np.power((box_cost_gain - box_cost_gain.min() + 1), args.crossover_mu)
    # prob = prob / prob.sum()
    crossover_idxes = np.random.choice(
        range(len(parent_a[0])), 
        size=int(args.crossover_prop*len(parent_a[0])), 
        replace=False,
        # p=prob,
    )
    item_idxes = np.where(parent_b[:, crossover_idxes]==1)[0]
    to_cleanup_boxes = list(set(np.where(parent_a[item_idxes]==1)[1]))
    parent_a[:, to_cleanup_boxes] = 0
    parent_a[item_idxes] = 0
    parent_a[:, crossover_idxes] = parent_b[:, crossover_idxes]
    numa_actions_a[:, to_cleanup_boxes] = 0
    numa_actions_a[item_idxes] = 0
    numa_actions_a[:, crossover_idxes] = numa_actions_b[:, crossover_idxes]
    not_placed_items = np.where(parent_a.sum(axis=-1) <= 0)[0]
    
    # pdb.set_trace()
    return _find_feasible_placement(ecs_env, parent_a, numa_actions_a, not_placed_items, args)
    

def _find_feasible_placement(ecs_env, placement, numa_actions, item_idxes, args):
    iter, place_flag = 0, True
    new_placement, new_numa_actions, new_score, num_moves = None, None, 0., 0.
    while place_flag and (iter < args.max_feasible_iter):
        iter += 1
        ecs_env.reset()
        ecs_env.set_cur_placement(dcp(placement), dcp(numa_actions))
        np.random.shuffle(item_idxes)
        place_flag = False
        for i in item_idxes:
            available_boxes = np.where(ecs_env.actions_available[i] > 0)[0]
            if len(available_boxes) <= 0:
                place_flag = True
                break
            # prob = -ecs_env.item_assign_box_cost[i, available_boxes]
            prob = ecs_env.get_item_rewards(i)[available_boxes]
            additional_val = 1 # abs(prob.max()) + 1
            prob = np.power((prob - prob.min() + additional_val), args.feasible_mu)
            prob = prob / prob.sum()
            target_box = np.random.choice(available_boxes, replace=False, size=1, p=prob)[0]
            ecs_env.step([i, target_box])
        if not place_flag:
            new_placement = dcp(ecs_env.cur_placement)
            new_numa_actions = dcp(ecs_env.cur_items_numa_actions)
            new_score, num_moves = ecs_env.get_cur_state_score(is_contain_migration=args.is_contain_migration, is_normcost=args.is_normcost)
            break
    return new_placement, new_numa_actions, new_score, num_moves


class GeneticAlgorithm:
    def __init__(self, args):
        self.args = args
        if os.path.exists(self.args.log_save_path):
            self.logger = open(os.path.join(self.args.log_save_path, self.args.data_path.split('/')[-1]+'.log'), 'wb')
        else:
            self.logger = None
        
    def _write_infos(self, info):
        if self.logger is not None:
            self.logger.write((info+'\n').encode())
            self.logger.flush()
        else:
            print(info)
    
    def init_population(self):
        population, numa_actions, scores, num_moves = [], [], [], []
        pool = Pool(processes=min(64, int(self.args.population_size*1.5)))
        results = []
        for i in range(int(self.args.population_size*1.5)):
            results.append(pool.apply_async(_random_move_to_new_placement, args=(i, self.args.seed+i, self.args, )))
        pool.close()
        pool.join()
        for r in results:
            r = r.get()
            population.append(r[0])
            numa_actions.append(r[1])
            scores.append(r[2])
            num_moves.append(r[3])
        # pdb.set_trace()
        return population, numa_actions, scores, num_moves
    
    def mutate_population(self, cur_population, cur_numa_actions, cur_scores, gen_i):
        mutate_size = int(self.args.population_size * self.args.mutation_rate)
        idxes = np.random.choice(range(self.args.population_size), size=mutate_size, replace=True)
        pool = Pool(processes=min(64, mutate_size))
        results = []
        for i in idxes:
            # self._mutate(cur_population[i], int(self.args.seed+i*10))
            results.append(pool.apply_async(_mutate, args=(cur_population[i], cur_numa_actions[i], int(self.args.seed+i*10+gen_i), self.args, )))
        pool.close()
        pool.join()
        
        mutation_population, mutation_numa_actions, mutation_scores, mutation_num_moves = [], [], [], []
        for r in results:
            r = r.get()
            if r[0] is not None:
                mutation_population.append(r[0])
                mutation_numa_actions.append(r[1])
                mutation_scores.append(r[2])
                mutation_num_moves.append(r[3])
        if len(mutation_population) <= 5:
            self.args.mutation_prop *= 0.8
        if len(mutation_population) >= 16:
            self.args.mutation_prop *= 1.2
        self.args.mutation_prop = np.clip(self.args.mutation_prop, 0.05, 0.2)
        return mutation_population, mutation_numa_actions, mutation_scores, mutation_num_moves
    
    def crossover_population(self, cur_population, cur_numa_actions, cur_scores, gen_i):
        # parents_b = list(set(range(len(cur_scores))) - set(parents_a))[:len(parents_a)]
        parents_a = np.arange(len(cur_scores), dtype=int)
        parents_b = np.random.permutation(parents_a)
        
        parents, parents_numa = [], []
        for a, b in zip(parents_a, parents_b):
            if np.random.uniform() < self.args.crossover_rate:
                parents.append([cur_population[a], cur_population[b]])
                parents_numa.append([cur_numa_actions[a], cur_numa_actions[b]])
        pool = Pool(processes=min(64, len(parents)))
        results = []
        # pdb.set_trace()
        for pi, p in enumerate(parents):
            # self._crossover(p[0].copy(), p[1].copy(), int(self.args.seed+pi*100+gen_i*10))
            results.append(pool.apply_async(_crossover, args=(p[0], p[1], parents_numa[pi][0], parents_numa[pi][1], int(self.args.seed+pi*100+gen_i*10), self.args, )))
        pool.close()
        pool.join()
        
        crossover_population, crossover_numa_actions, crossover_scores, crossover_num_moves = [], [], [], []
        for r in results:
            r = r.get()
            if r[0] is not None:
                crossover_population.append(r[0])
                crossover_numa_actions.append(r[1])
                crossover_scores.append(r[2])
                crossover_num_moves.append(r[3])
        return crossover_population, crossover_numa_actions, crossover_scores, crossover_num_moves
    
    def select_population(self, population, numa_actions, scores, num_moves):
        if self.args.is_selection_random:
            prob = - np.array(scores)
            prob = np.power((prob - prob.min() + 1), self.args.selection_mu)
            prob = prob / prob.sum()
            idxes = np.random.choice(range(len(scores)), replace=False, size=self.args.population_size, p=prob)
        else:
            idxes = np.argsort(scores)[:self.args.population_size]
        np.random.shuffle(idxes)
        cur_population, cur_scores, cur_num_moves = np.array(population)[idxes], np.array(scores)[idxes], np.array(num_moves)[idxes]
        cur_numa_actions = np.array(numa_actions)[idxes]
        best_idx = np.argmin(cur_scores)
        return cur_population, cur_numa_actions, cur_scores, cur_num_moves, cur_population[best_idx], cur_numa_actions[best_idx], cur_scores[best_idx], cur_num_moves[best_idx]
    
    def run_genetic_algorithm(self):
        self._write_infos(f"Start Running Genetic Algorithm on {self.args.data_path}...")
        start = time.time()
        new_population, new_numa_actions, new_scores, new_num_moves = self.init_population()
        best_placement, best_numa_actions, best_score = None, None, np.inf
        self._write_infos(f"Finish Initialize Population {len(new_population)}, Elapsed Time: {time.time()-start}s!")
        
        blocked_iter = 0
        for gi in range(self.args.generations):
            st = time.time()
            cur_population, cur_numa_actions, cur_scores, cur_num_moves, cur_best_placement, cur_best_numa_actions, cur_best_score, cur_best_num_move = \
                self.select_population(new_population, new_numa_actions, new_scores, new_num_moves)
            relative_gap = (best_score - cur_best_score) / (abs(cur_best_score) + 1e-5)
            if cur_best_score < best_score:
                best_placement, best_numa_actions, best_score = cur_best_placement, cur_best_numa_actions.reshape(len(cur_best_numa_actions), -1), cur_best_score
                sparse.save_npz(os.path.join(self.args.save_path, self.args.data_path.split('/')[-1]+'_ga_placement.npz'), sparse.csr_matrix(best_placement))
                sparse.save_npz(os.path.join(self.args.save_path, self.args.data_path.split('/')[-1]+'_ga_numa.npz'), sparse.csr_matrix(best_numa_actions))
                self._write_infos(f"* Find Better Placement, Current Best Score: {best_score}, Current Num Move: {cur_best_num_move}, Relative Gap {relative_gap}!")
            if (relative_gap < self.args.rel_gap):
                blocked_iter += 1
                if blocked_iter >= self.args.max_blocked_iter:
                    self._write_infos(f"! Relative Gap {relative_gap} is small, Stop Iteration!")
                    break
            else:
                blocked_iter = 0
            if (time.time() - start) > self.args.time_limit:
                self._write_infos(f"! Time Limit {self.args.time_limit}s Exceed!")
                break
            crossover_placement, crossover_numa_actions, crossover_scores, crossover_num_moves = self.crossover_population(cur_population, cur_numa_actions, cur_scores, gi)
            if len(crossover_placement) > 0:
                new_population = np.concatenate([cur_population, crossover_placement], axis=0)
                new_numa_actions = np.concatenate([cur_numa_actions, crossover_numa_actions], axis=0)
                new_scores = np.concatenate([cur_scores, crossover_scores], axis=0)
                new_num_moves = np.concatenate([cur_num_moves, crossover_num_moves], axis=0)
            else:
                new_population, new_numa_actions, new_scores, new_num_moves = cur_population, cur_numa_actions, cur_scores, cur_num_moves
            mutate_placement, mutate_numa_actions, mutate_scores, mutate_num_moves = self.mutate_population(cur_population, cur_numa_actions, cur_scores, gi)
            if len(mutate_placement) > 0:
                new_population = np.concatenate([new_population, mutate_placement], axis=0)
                new_numa_actions = np.concatenate([new_numa_actions, mutate_numa_actions], axis=0)
                new_scores = np.concatenate([new_scores, mutate_scores], axis=0)
                new_num_moves = np.concatenate([new_num_moves, mutate_num_moves], axis=0)
            self._write_infos(f"@Epoch{gi}, Crossover Population {len(crossover_scores)}, Mutation Population {len(mutate_scores)}, Spend Time {time.time()-st}s!")
            # pdb.set_trace()
        # sparse.save_npz(os.path.join(self.args.save_path, self.args.data_path.split('/')[-1]+'_ga_placement.npz'), sparse.csr_matrix(best_placement))
        self._write_infos(f"Finish Solved {args.data_path} by GeneticAlgorithm!, Best Cost: {best_score}, Total Time: {time.time()-start}s!")
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='../../data/ecs_data')
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=15793)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--rel_gap", type=float, default=1e-3)
    parser.add_argument("--max_blocked_iter", type=int, default=10)
    parser.add_argument("--init_mu", type=float, default=1.)
    parser.add_argument("--is_selection_random", action='store_true', default=False)
    parser.add_argument("--selection_mu", type=float, default=0.5)
    
    parser.add_argument("--mutation_rate", type=float, default=1.)
    parser.add_argument("--mutation_prop", type=float, default=0.1)
    parser.add_argument("--max_feasible_iter", type=int, default=3)
    parser.add_argument("--feasible_mu", type=float, default=0.5)
    parser.add_argument("--mutation_mu", type=float, default=0.5)
    
    parser.add_argument("--crossover_rate", type=float, default=0.9)
    parser.add_argument("--crossover_prop", type=float, default=0.02)
    parser.add_argument("--crossover_mu", type=float, default=0.5)
    
    parser.add_argument("--time_limit", type=int, default=7200)
    
    parser.add_argument("--is_contain_migration", action='store_false', default=True)
    parser.add_argument("--is_normcost", action='store_true', default=False)
    
    parser.add_argument("--save_path", type=str, default='../../results/ecs_metaheuristic_ga/placement_ecs_data')
    parser.add_argument("--log_save_path", type=str, default='../../results/ecs_metaheuristic_ga/logs_ecs_data')
    
    args = parser.parse_args()
    args.save_path = f"{args.save_path}{'_mig' if args.is_contain_migration else ''}{'_normcost' if args.is_normcost else ''}"
    args.log_save_path = f"{args.log_save_path}{'_mig' if args.is_contain_migration else ''}{'_normcost' if args.is_normcost else ''}"
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_save_path, exist_ok=True)
    
    data_root_path = args.data_path
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    with open('../../data/ecs_data_classification.json', 'r') as f:
        data_classified = json.load(f)
        data_list = [data_classified[f"type_{i}"] for i in range(len(data_classified))]
    
    for datas in data_list:
        for data in datas:
            if not os.path.exists(os.path.join(args.save_path, f"{data}_ga_placement.npz")):
                args.data_path = f"../../data/ecs_data/{data}"
                ga = GeneticAlgorithm(args=args)
                ga.run_genetic_algorithm()
