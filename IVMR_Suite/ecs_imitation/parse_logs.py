import os
import pandas as pd
import numpy as np
import pdb


def parse_logs_to_results(logs_root_path, save_path=None):
    results = []
    for file in os.listdir(logs_root_path):
        r = [file] + [0] * 16
        with open(os.path.join(logs_root_path, file), 'r') as f:
            lines = f.readlines()[::-1]
            for l in lines:
                if l.startswith('Init Score:'):
                    r[1] = float(l.split('Init Score: ')[1].split(',')[0].strip())
                    r[2] = float(l.split('Final Score: ')[1].split(',')[0].strip())
                    r[3] = int(l.split('Num Action: ')[1].split(',')[0].strip())
                    r[4] = float(l.split('Total Reward: ')[1].split(',')[0].strip())
                    r[5] = float(l.split('Total Real Reward: ')[1].split('!')[0].strip())
                    r[6] = float(l.split('Elapsed Time: ')[1].split('s')[0].strip())
                elif l.startswith('Initial State Vio Cost:'):
                    r[7] = float(l.split('Initial State Vio Cost: ')[1].split(',')[0])
                    r[8] = int(l.split('Mutex Box: ')[1].split(',')[0])
                    r[9] = int(l.split('Mutex Item: ')[1].split(',')[0])
                    r[10] = int(l.split('Resource: ')[1].split(',')[0])
                    r[11] = int(l.split('Numa: ')[1].split('!')[0])
                elif l.startswith('Final State Vio Cost:'):
                    r[12] = float(l.split('Final State Vio Cost: ')[1].split(',')[0])
                    r[13] = int(l.split('Mutex Box: ')[1].split(',')[0])
                    r[14] = int(l.split('Mutex Item: ')[1].split(',')[0])
                    r[15] = int(l.split('Resource: ')[1].split(',')[0])
                    r[16] = int(l.split('Numa: ')[1].split('!')[0])
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'init_score', 'final_score', 'num_action',
                                             'total_reward', 'total_real_reward', 'time',
                                             'init_vio_cost', 'init_mutex_box', 'init_mutex_item', 'init_res', 'init_numa',
                                             'f_vio_cost', 'f_mutex_box', 'f_mutex_item', 'f_res', 'f_numa'])
    results['vio_cost'] = results['init_vio_cost'] - results['f_vio_cost']
    results['mutex_box'] = results['init_mutex_box'] - results['f_mutex_box']
    results['mutex_item'] = results['init_mutex_item'] - results['f_mutex_item']
    results['vio_res'] = results['init_res'] - results['f_res']
    results['vio_numa'] = results['init_numa'] - results['f_numa']
    results['total_vio_num'] = results['mutex_box'] + results['mutex_item'] + results['vio_res'] + results['vio_numa']
    # pdb.set_trace()
    if save_path is not None:
        results.to_csv(save_path, sep=',', header=True, index=False)
    return results
    

if __name__ == '__main__':
    # type_list = list(range(9)) + [-1]
    type_list = [-1]
    is_merge_results = True
    full_results, full_results_name = [], 'ecs_task_v5_v0_t0-1'
    for typei in type_list:
        # model_version = f'ecs_task_v5_v0_t1_model_best_ecs_task_v5_v0_t0'
        model_version = f'ecs_task_stock_box_v0_t{typei}'
        model_name = 'model_best'
        logs_root_path = f'../results/ecs_imitation/results/eval_logs_with_live_migrate_limit_migrated/{model_version}_{model_name}'
        save_root_path = '../results/ecs_imitation/results/eval_results_with_live_migrate_limit_migrated'
        os.makedirs(save_root_path, exist_ok=True)
        save_path = f'{save_root_path}/{model_version}_{model_name}.csv'
        result = parse_logs_to_results(logs_root_path=logs_root_path, save_path=save_path)
        if typei >= 0:
            full_results.append(result)
    
    if (len(full_results) > 1) and is_merge_results:
        full_results = pd.concat(full_results, axis=0)
        full_results.to_csv(os.path.join(save_root_path, f'{full_results_name}_{model_name}.csv'), sep=',', header=True, index=False)
