import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from ecs_opt.opt_get_move_seq import MoveSeqGenerator
import time


if __name__ == '__main__':
    task_type = 'ecs_data'
    data_root_path = f'../data/{task_type}'
    is_filter_unmovable, is_process_numa = True, True
    logger_root_path = f'../results/ecs_opt/opt_logs/{task_type}_addmig1.0_move_seq'
    os.makedirs(logger_root_path, exist_ok=True)
    solution = None
    sol_root_save_path = f'../results/ecs_opt/opt_sols/{task_type}_addmig1.0'
    is_solve_cycle, is_undo_neg, is_sorted_moves = True, True, True
    
    def get_move_seq(data_name):
        if os.path.exists(os.path.join(logger_root_path, f"{data_name}.log")):
            print(f"Already solved {data_name}")
            return
        generator = MoveSeqGenerator(os.path.join(data_root_path, data_name), is_filter_unmovable, is_process_numa, 
                                     os.path.join(logger_root_path, f"{data_name}.log"), solution)
        generator.opt_solve_ecs(os.path.join(sol_root_save_path, f"{data_name}.mst"), is_solve_cycle, is_undo_neg, is_sorted_moves)
    
    sols_list = os.listdir(sol_root_save_path)
    
    pool = Pool(processes=4)
    for sol in sols_list:
        pool.apply_async(get_move_seq, args=(sol.split('.')[0],))
    pool.close()
    pool.join()
    
