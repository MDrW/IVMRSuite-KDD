import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from ecs_opt.cplex_model_and_solve_binary import ECSCplexOptModel
import time
import json
from argparse import ArgumentParser


def _model_and_solve_task(file_name, root_path, save_root_path, sol_save_root_path, logger_root_path, add_migration_cost=0, 
                          is_remove_unmovable=True, is_gzip=True, is_solve=True, 
                          cplex_params={'timelimit': 2000, 'threads': 32, 'mip.tolerances.mipgap': 1e-2}):
    if os.path.exists(os.path.join(logger_root_path, file_name+'.log')):
        return None
    st = time.time()
    ecs_opt_model = ECSCplexOptModel(
        data_path=os.path.join(root_path, file_name),
        model_save_path=os.path.join(save_root_path, file_name+'.lp'),
        logger_path=os.path.join(logger_root_path, file_name+'.log'),
        cplex_params=cplex_params,
        add_migration_cost=add_migration_cost,
        is_remove_unmovable=is_remove_unmovable,
    )
        
    ecs_opt_model.model()
    if is_gzip and (not os.path.exists(os.path.join(save_root_path, file_name+'.lp.gz'))) and os.path.exists(os.path.join(save_root_path, file_name+'.lp')):
        os.system(f"gzip {os.path.join(save_root_path, file_name)}.lp")
    if is_solve and (not os.path.exists(os.path.join(sol_save_root_path, file_name+'.mst'))):
        ecs_opt_model.solve(sol_save_path=sol_save_root_path)
    end = time.time()
    info = f"{file_name} is Solved! Total Elapsed Time: {end-st}s."
    ecs_opt_model._write_info(info)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_type", type=str, default="ecs_data")
    parser.add_argument("--is_solve", action='store_false', default=True)
    parser.add_argument("--num_threads", type=int, default=2)
    parser.add_argument("--is_gzip", action='store_false', default=True)
    parser.add_argument("--is_remove_unmovable", action='store_false', default=True)
    parser.add_argument("--add_migration_cost", type=float, default=1.)
    args = parser.parse_args()
    
    data_type = args.data_type
    is_solve = args.is_solve
    num_threads = args.num_threads
    is_gzip = args.is_gzip
    add_migration_cost = args.add_migration_cost
    
    root_path = f"../data/{data_type}"
    save_root_path = f'../results/ecs_opt/opt_models/{data_type}_addmig{add_migration_cost}'
    sol_save_root_path = f'../results/ecs_opt/opt_sols/{data_type}_addmig{add_migration_cost}'
    logger_root_path = f'../results/ecs_opt/logs_solve/{data_type}_addmig{add_migration_cost}_model_and_solve'
    os.makedirs(save_root_path, exist_ok=True)
    os.makedirs(logger_root_path, exist_ok=True)
    os.makedirs(sol_save_root_path, exist_ok=True)
    
    with open(f'../data/ecs_data_classification.json', 'r') as f:
        data_dict = json.load(f)
        data_list = []
        for k in [f'type_{i}' for i in range(len(data_dict))]:
            data_list.append(data_dict[k])
    print(f"Start Processing {len(data_list)} Files in {data_type}...")
    
    # cplex_params_list = [
    #     # {'timelimit': 3600, 'threads': 32, 'mip.tolerances.mipgap': 0.01},
    #     # {'timelimit': 7200, 'threads': 32, 'mip.tolerances.mipgap': 0.01},
    #     # {'timelimit': 10800, 'threads': 32, 'mip.tolerances.mipgap': 0.01},
    #     # {'timelimit': 10800, 'threads': 32, 'mip.tolerances.mipgap': 0.01},
    # ]
    cplex_params_list = [{'timelimit': None, 'threads': 32, 'mip.tolerances.mipgap': 0.01} for _ in range(len(data_list))]
    
    st = time.time()
    pool = Pool(processes=num_threads)
    for datas, cplex_params in zip(data_list, cplex_params_list):
        for file in datas:
            pool.apply_async(
                _model_and_solve_task, 
                args=(str(file), root_path, save_root_path, sol_save_root_path, logger_root_path, add_migration_cost, args.is_remove_unmovable, is_gzip, is_solve, cplex_params, ), 
                error_callback=lambda e: print(e))
    pool.close()
    pool.join()

    end = time.time()
    
    print(f"Finish Processed {data_type}, Total Elapsed Time: {end - st}s.")
