import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ecs_env.ecs_base_env import ECSEnvironment
import time
import random


def greedy_policy(ecs_env: ECSEnvironment, logger=None, seed=15937, max_bad_action_count=0):
    np.random.seed(seed)
    random.seed(seed)
    state = ecs_env.reset()
    s = time.time()
    available_actions = ecs_env.get_available_actions()
    item_count, box_count = available_actions.shape
    reward_matrix = ecs_env._get_reward_batch(range(box_count))
    reward_matrix[(1-available_actions).astype(bool)] = -1e99
    e = time.time()
    print(f"Prepare Time: {e-s}s.")
    total_reward, action_list = 0, []
    bad_action_count = 0
    while available_actions.sum() > 0:
        s1 = time.time()
        ind_choice = np.argmax(reward_matrix)
        ind_choice = np.unravel_index(ind_choice, reward_matrix.shape)
        value, action = reward_matrix[ind_choice], list(ind_choice)
        if value <= 0:
            bad_action_count += 1
            if bad_action_count > max_bad_action_count:
                break
        else:
            bad_action_count = 0
        item_cur_box = ecs_env.get_item_cur_box(action[0])
        s2 = time.time()
        _, reward, done, infos = ecs_env.step(list(action))
        e2 = time.time()
        move_count, action_info = infos["move_count"], infos["action_info"]
        if logger is None:
            print(f"Step {move_count}: {reward}, {value}, {done}, {action_info}, {e2-s2}s, {e2-s1}s.")
        else:
            logger.write(f"Step {move_count}: {reward}, {done}, {action_info}, {e2-s2}s, {e2-s1}s.\n".encode())
            logger.flush()
        total_reward += reward
        action_list.append(list(action))
        
        available_actions = ecs_env.get_available_actions()
        box_update = [int(item_cur_box), action[1]]
        reward_update = ecs_env._get_reward_batch(box_update)
        reward_matrix[:, box_update] = reward_update
        reward_matrix[(1-available_actions).astype(bool)] = -1e99
            
        if done:
            break
    if logger is None:
        print("Actions Count:", len(action_list), "Reward:", total_reward)
    else:
        logger.write(f"Actions Count: {len(action_list)}, Reward: {total_reward}\n".encode())
        logger.flush()
    return action_list


if __name__ == '__main__':
    data_path = '../data/ecs_data/0'
    # data_path = './data/nocons_finite_hard/819368'
    s1 = time.time()
    env = ECSEnvironment(data_path=data_path, is_limited_count=True, is_filter_unmovable=True)
    init_score = env.get_cur_state_score()
    e1 = time.time()
    
    print("Initial State Score:", init_score, e1-s1)
    action_list = greedy_policy(ecs_env=env)
    e2 = time.time()
    
    eval_reward, cost_infos, inv_num = env.eval_move_sequence(action_list)
    e3 = time.time()
    des_score = env.get_cur_state_score()
    e4 = time.time()
    print("Eval Reward:", eval_reward)
    print("Migration Cost:", cost_infos['migration_cost'], "Finish Score:", des_score, 
          "Total Eval Score:", des_score+cost_infos['migration_cost']-init_score)
    print("Elapsed Time:", e2-e1, e3-e2, e4-e3)
