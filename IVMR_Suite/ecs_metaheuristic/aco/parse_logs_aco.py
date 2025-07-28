import numpy as np
import pandas as pd
import os
import pdb


def parse_placement_logs(logger_root_path, data_path='../../data/ecs_data'):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            res = [0] * 3
            for line in f:
                if line.startswith('Start Running'):
                    res[0] = os.path.join(data_path, (line.split(' on ')[1].split('...')[0]).split('/')[-1])
                    # pdb.set_trace()
                elif line.startswith('Finish Solved'):
                    res[1] = float(line.split('Best Cost: ')[1].split(',')[0].strip())
                    res[2] = float(line.split('Total Time: ')[1].split('s')[0].strip())
            results.append(res)
    results = pd.DataFrame(results, columns=['name', 'best_cost', 'placement_total_time'])
    return results


def parse_move2seq_logs(logger_root_path, data_path='../../data/ecs_data'):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            r = [0] * 20
            for line in f:
                if line.startswith('Initial Score: '):
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
                    r[18] += 1
                elif line.startswith('[Undo] Step'):
                    r[19] += 1
                elif line.startswith('After Sorting Moves'):
                    r[11] = int(line.split('Move Count: ')[1].split(',')[0].strip())
                    r[12] = float(line.split('Total Reward: ')[1].split(',')[0].strip())
                    r[13] = float(line.split('Final Score: ')[1].split(',')[0].strip())
                    r[14] = int(line.split('Invalid Num: ')[1].split(',')[0].strip())
                    r[15] = float(line.split('Eval Reward: ')[1].split(',')[0].strip())
                    # r[16] = float(line.split('Eval Final Score:')[1].split(',')[0].strip())
                    r[17] = float(line.split('Elapsed Time: ')[1].split('s')[0].strip())
            if r[0] == 0:
                r[0] = f"{data_path}/{file.split('.')[0]}"
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'init_cost', 'opt_score', 'opt_move_count', 'real_move_count', 'total_reward',
                                             'final_score', 'invalid_num', 'eval_reward', 'eval_final_score', 'without_sort_time',
                                             'sorted_move_count', 'sorted_total_reward', 'sorted_final_score',
                                             'sorted_invalid_num', 'sorted_eval_reward', 'sorted_eval_final_score', 'sorted_total_time',
                                             'num_break_cycle', 'num_undo'])
    results.loc[results['real_move_count']<=0, ['sorted_total_reward', 'sorted_final_score', 'sorted_eval_reward']] = \
        results.loc[results['real_move_count']<=0, ['total_reward', 'final_score', 'eval_reward']].values
    return results


def parse_logger(placement_root_path, move2seq_root_path, result_save_path, data_path='../../data/ecs_data'):
    placement_results = parse_placement_logs(placement_root_path, data_path=data_path)
    #pdb.set_trace()
    move2seq_results = parse_move2seq_logs(move2seq_root_path, data_path=data_path)
    # pdb.set_trace()
    results = pd.merge(placement_results, move2seq_results, on='name', how='inner')
    #pdb.set_trace()
    results['total_time'] = results['placement_total_time'] + results['sorted_total_time']
    results.to_csv(result_save_path, sep=',', header=True, index=False)
    return results


if __name__ == '__main__':
    root_path = '../../results/ecs_metaheuristic_aco'
    placement_root_path = f'{root_path}/logs_ecs_data'
    move2seq_root_path = f'{root_path}/moveseq_ecs_data'
    save_path = f'{root_path}/results/aco_ecs_data.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data_path = f'../../data/ecs_data'
    parse_logger(
        placement_root_path=placement_root_path,
        move2seq_root_path=move2seq_root_path,
        result_save_path=save_path,
        data_path=data_path,
    )
