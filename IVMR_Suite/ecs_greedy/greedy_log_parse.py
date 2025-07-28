import pandas as pd
import os


def parse_logger(logger_root_path, result_save_path):
    results = []
    for file in os.listdir(logger_root_path):
        with open(os.path.join(logger_root_path, file), 'r') as f:
            r = [0] * 8
            for line in f:
                if line.startswith('Start Running'):
                    r[0] = line.split('Start Running ')[1].split('...')[0]
                elif line.startswith('Initial State Score: '):
                    r[4] = float(line.split('Initial State Score: ')[1])
                elif line.startswith('Actions Count: '):
                    r[2] = int(line.split('Actions Count: ')[1].split(',')[0])
                elif line.startswith('Eval Reward: '):
                    r[1] = float(line.split('Eval Reward: ')[1])
                elif line.startswith('Migration Cost: '):
                    r[3] = float(line.split('Migration Cost: ')[1].split(',')[0])
                    r[5] = float(line.split('Finish Score: ')[1].split(',')[0])
                    r[6] = float(line.split('Total Eval Score: ')[1])
                elif line.startswith('Elapsed Time: '):
                    r[7] = float(line.split('Elapsed Time: ')[1].split('s')[0])
            results.append(r)
    results = pd.DataFrame(results, columns=['name', 'eval_reward', 'action_count', 'migration_cost',
                                             'init_cost', 'final_cost', 'score', 'time'])
    results.to_csv(result_save_path, sep=',', header=True, index=False)
    
    
if __name__ == '__main__':
    logger_root_path = './logs/greedy_ecs_data'
    result_save_path = './results/greedy_ecs_data.csv'
    parse_logger(logger_root_path=logger_root_path, result_save_path=result_save_path)
    