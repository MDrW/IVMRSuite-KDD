import pandas as pd
import os
import pdb


def parse_modeling_results(logger_root_path, result_save_path=None):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            r = [0] * 10
            for line in f:
                if line.startswith('Start Constructing Model '):
                    r[0] = line.split('Start Constructing Model ')[1].split('...')[0]
                elif line.startswith('Variables are Defined! Elapsed Time: '):
                    r[1] = float(line.split('Variables are Defined! Elapsed Time: ')[1].split('s')[0].strip())
                elif line.startswith('Objective is Defined! Elapsed Time: '):
                    r[2] = float(line.split('Objective is Defined! Elapsed Time: ')[1].split('s')[0].strip())
                elif line.startswith('   * Constraints: Item Can Migrate are Defined! Spend Time: '):
                    r[3] = float(line.split('   * Constraints: Item Can Migrate are Defined! Spend Time: ')[1].split('s')[0].strip())
                elif line.startswith('   * Constraints: Item Placement Limitation are Defined! Spend Time: '):
                    r[4] = float(line.split('   * Constraints: Item Placement Limitation are Defined! Spend Time: ')[1].split('s')[0].strip())
                elif line.startswith('   * Constraints: Item Mutex Box are Defined! Spend Time: '):
                    r[5] = float(line.split('   * Constraints: Item Mutex Box are Defined! Spend Time: ')[1].split('s')[0].strip())
                elif line.startswith('   * Constraints: Box Simple Resources Sat are Defined! Spend Time: '):
                    r[6] = float(line.split('   * Constraints: Box Simple Resources Sat are Defined! Spend Time: ')[1].split('s')[0].strip())
                elif line.startswith('Constraints are Defined! Elapsed Time: '):
                    r[7] = float(line.split('Constraints are Defined! Elapsed Time: ')[1].split('s')[0].strip())
                elif line.startswith('Write Model to '):
                    r[8] = float(line.split('Total Elapsed Time: ')[1].split('s')[0].strip())
                elif 'Constructed! Total Elapsed Time: ' in line:
                    r[9] = float(line.split('Constructed! Total Elapsed Time: ')[1].split('s')[0].strip()) - r[8]
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'vars_time', 'obj_time', 'migrate_cons_time',
                                             'placement_cons_time', 'mutex_box_cons_time', 'simple_res_cons_time',
                                             'cons_time', 'write_time', 'model_total_time(without_write)'])
    results['name'] = results['name'].astype(str).apply(lambda x: x.split('/')[-1])
    if result_save_path is not None:
        results.to_csv(result_save_path, sep=',', header=True, index=False)
    return results


def parse_cplex_solve_results(logger_root_path, result_save_path=None):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            r = [0] * 4
            for line in f:
                if line.startswith('Start Solving'):
                    r[0] = line.split('Start Solving ')[1].split('...')[0]
                elif line.startswith('Total (root+branch&cut) ='):
                    r[1] = float(line.split('Total (root+branch&cut) =')[1].split('sec')[0].strip())
                elif line.startswith('The Optimal Objective: '):
                    r[2] = float(line.split('The Optimal Objective: ')[1].split('!')[0].strip())
                elif line.startswith('gap     = '):
                    r[3] = float(line.split('gap     = ')[1].split('%')[0].strip()) / 100
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'opt_solve_time', 'opt_objective', 'relative_gap'])
    results['name'] = results['name'].astype(str).apply(lambda x: x.split('/')[-1])
    if result_save_path is not None:
        results.to_csv(result_save_path, sep=',', header=True, index=False)
    return results


def parse_solve_results(logger_root_path, result_save_path=None, data_path='../data/nocons_finite_easy'):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            r = [0] * 35
            for line in f:
                if line.startswith('Start Opt Solving'):
                    r[0] = line.split('Start Opt Solving ')[1].split('...')[0]
                elif line.startswith('Initial Score: '):
                    r[1] = float(line.split('Initial Score: ')[1].split(',')[0].strip())
                    r[2] = float(line.split('Optimal Score: ')[1].split(',')[0].strip())
                    r[3] = int(line.split('Optimal Move Count: ')[1].split('!')[0].strip())
                elif line.startswith('Real Move Count: '):
                    r[4] = int(line.split('Real Move Count: ')[1].split(',')[0].strip())
                    r[5] = float(line.split('Total Reward: ')[1].split(',')[0].strip())
                    r[6] = float(line.split('Final Score: ')[1].split(',')[0].strip())
                    r[7] = int(line.split('Invalid Num: ')[1].split(',')[0].strip())
                    r[8] = float(line.split('Eval Reward: ')[1].split(',')[0].strip())
                    r[9] = float(line.split('Eval Final Score: ')[1].split(',')[0].strip())
                    r[10] = float(line.split('Elapsed Time: ')[1].split('s')[0].strip())
                elif line.startswith('[Break Cycle] Step'):
                    r[17] += 1
                elif line.startswith('[Undo] Step'):
                    r[18] += 1
                elif line.startswith('Finish Loaded Solution from'):
                    r[19] = float(line.split('Elapsed Time ')[1].split('s')[0].strip())
                elif line.startswith('After Sorting Moves, Move Count'):
                    r[11] = int(line.split('Move Count: ')[1].split(',')[0].strip())
                    r[12] = float(line.split('Total Reward: ')[1].split(',')[0].strip())
                    r[13] = float(line.split('Final Score: ')[1].split(',')[0].strip())
                    r[14] = int(line.split('Invalid Num: ')[1].split(',')[0].strip())
                    r[15] = float(line.split('Eval Reward: ')[1].split(',')[0].strip())
                    r[16] = float(line.split('Elapsed Time: ')[1].split('s')[0].strip())
                elif line.startswith('Initial State Vio Cost:'):
                    r[20] = float(line.split('Initial State Vio Cost: ')[1].split(',')[0])
                    r[21] = int(line.split('Mutex Box: ')[1].split(',')[0])
                    r[22] = int(line.split('Mutex Item: ')[1].split(',')[0])
                    r[23] = int(line.split('Resource: ')[1].split(',')[0])
                    r[24] = int(line.split('Numa: ')[1].split('!')[0])
                elif line.startswith('Final Vio Cost: '):
                    r[25] = float(line.split('Final Vio Cost: ')[1].split(',')[0])
                    r[26] = int(line.split('Mutex Box: ')[1].split(',')[0])
                    r[27] = int(line.split('Mutex Item: ')[1].split(',')[0])
                    r[28] = int(line.split('Resource: ')[1].split(',')[0])
                    r[29] = int(line.split('Numa: ')[1].split('!')[0])
                elif line.startswith('Opt State Vio Cost:'):
                    r[30] = float(line.split('Opt State Vio Cost: ')[1].split(',')[0])
                    r[31] = int(line.split('Mutex Box: ')[1].split(',')[0])
                    r[32] = int(line.split('Mutex Item: ')[1].split(',')[0])
                    r[33] = int(line.split('Resource: ')[1].split(',')[0])
                    r[34] = int(line.split('Numa: ')[1].split('!')[0])
            # if r[0] == 0:
            r[0] = f"{data_path}/{file.split('.')[0]}"
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'init_cost', 'opt_score', 'opt_move_count', 'real_move_count', 'total_reward',
                                             'final_score', 'invalid_num', 'eval_reward', 'eval_final_score', 'without_sort_time',
                                             'sorted_move_count', 'sorted_total_reward', 'sorted_final_score',
                                             'sorted_invalid_num', 'sorted_eval_reward', 'sorted_total_time',
                                             'num_break_cycle', 'num_undo', 'load_sol_time',
                                             'init_vio_cost', 'init_mutex_box', 'init_mutex_item', 'init_res', 'init_numa',
                                             'f_vio_cost', 'f_mutex_box', 'f_mutex_item', 'f_res', 'f_numa',
                                             'opt_vio_cost', 'opt_mutex_box', 'opt_mutex_item', 'opt_res', 'opt_numa'])
    results.loc[results['real_move_count']<=0, ['sorted_total_reward', 'sorted_final_score', 'sorted_eval_reward']] = \
        results.loc[results['real_move_count']<=0, ['total_reward', 'final_score', 'eval_reward']].values
    results['vio_cost'] = results['init_vio_cost'] - results['f_vio_cost']
    results['mutex_box'] = results['init_mutex_box'] - results['f_mutex_box']
    results['mutex_item'] = results['init_mutex_item'] - results['f_mutex_item']
    results['vio_res'] = results['init_res'] - results['f_res']
    results['vio_numa'] = results['init_numa'] - results['f_numa']
    results['total_vio_num'] = results['mutex_box'] + results['mutex_item'] + results['vio_res'] + results['vio_numa']
    results['name'] = results['name'].astype(str).apply(lambda x: x.split('/')[-1])
    if result_save_path is not None:
        results.to_csv(result_save_path, sep=',', header=True, index=False)
    return results


def parse_logger(model_root_path, cplex_root_path, solve_root_path, result_save_path, data_path='../data/nocons_finite_easy'):
    model_results = parse_modeling_results(model_root_path, result_save_path=None)
    cplex_results = parse_cplex_solve_results(cplex_root_path, result_save_path=None)
    solve_results = parse_solve_results(solve_root_path, result_save_path=None, data_path=data_path)
    # pdb.set_trace()
    results = pd.merge(model_results, cplex_results, on=['name'], how='inner')
    results = pd.merge(results, solve_results, on=['name'], how='inner')
    results['total_time'] = results['model_total_time(without_write)'] + results['opt_solve_time'] + results['without_sort_time'] - results['load_sol_time']
    results.to_csv(result_save_path, sep=',', header=True, index=False)
    return results
    
    
if __name__ == '__main__':
    solve_data_type = 'ecs_data'
    root_path = f'../data/{solve_data_type}'
    model_root_path = f'../results/ecs_opt/logs_solve/{solve_data_type}_addmig1.0_model_and_solve'
    cplex_root_path = f'../results/ecs_opt/logs_solve/{solve_data_type}_addmig1.0_model_and_solve'
    solve_root_path = f'../results/ecs_opt/opt_logs/{solve_data_type}_addmig1.0_move_seq'
    result_save_path = f'../results/ecs_opt/results/opt_{solve_data_type}.csv'
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)
    parse_logger(
        model_root_path=model_root_path,
        cplex_root_path=cplex_root_path,
        solve_root_path=solve_root_path,
        result_save_path=result_save_path,
        data_path=root_path,
    )
    