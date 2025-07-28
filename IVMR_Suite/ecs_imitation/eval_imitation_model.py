import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import torch
import torch_geometric
import random
import time
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
import numpy as np
import matplotlib.pyplot as plt
from ecs_imitation.gnn_model import GNNPolicy
from ecs_imitation.graph_dataset import GraphDataset
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
import pdb
from scipy import sparse
from ecs_env.ecs_env_binary_wrapper import ECSRawDenseEnvironment
from multiprocessing import Pool
from copy import deepcopy as dcp
import json
import pandas as pd


def _check_action_can_undo(action, env, undo_env, check_action_list):
    undo_flag = True
    simulation_env = dcp(env)
    _, _, _, info = simulation_env.undo_step(list([action[0], action[2], action[1]]))
    if not info['action_available']:
        undo_flag = False
    else:
        if check_action_list:
            undo_simu_env = dcp(undo_env)
            for ci, ck_act in enumerate(check_action_list):
                _, _, d, info = undo_simu_env.step([ck_act[0], ck_act[2]])
                if not info['action_available']:
                    undo_flag = False
                    break
                # if (ci != len(check_action_list)-1) and d:
                #     undo_flag = False
                #     break
    return undo_flag


@torch.no_grad()
def eval_one_epoch(args):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    log_file = open(args.log_save_path, 'wb')
    log_file.write(f"Start Eval {args.data_path}...\n".encode())
    log_file.flush()
    
    start = time.time()
    dataset = GraphDataset([args.data_path])
    
    ecs_env = ECSRawDenseEnvironment(data_path=args.data_path, is_limited_count=args.is_limited_count,
                                     is_filter_unmovable=args.is_filter_unmovable, is_dense_items=args.is_dense_items,
                                     is_dense_boxes=args.is_dense_boxes, is_state_merge_placement=args.is_state_merge_placement, is_process_numa=args.is_process_numa)
    is_store_pre, pre_idx, ecs_env_pre, reward_pre, real_reward_pre = False, -1, None, 0, 0
    state = ecs_env.reset()
    action_available = np.asarray(state[-2].todense())
    if action_available.sum() <= 0:
        return None
    graph = dataset.process_and_get_graph(state)
    init_score = ecs_env.get_cur_state_score()
    init_vio_cost, init_vio_info = ecs_env.get_cur_state_vio_cost()
    st1 = time.time()
    predict_model = GNNPolicy(
        n_item_feats=graph.n_item_feats, 
        n_box_feats=graph.n_box_feats, 
        n_edge_feats=graph.n_edge_feats,
        emb_size_list=args.emb_size_list,
        out_emb_size_list=args.out_emb_size_list,
        n_gnn_layers=args.n_gnn_layers,
        is_use_full_infos=args.is_use_full_infos,
        is_gnn_resnet=args.is_gnn_resnet, 
        graph_type=args.graph_type,
        normalization=args.normalization,
        activate_func=args.activate_func,
        is_box_outfeats_use_item=args.is_box_outfeats_use_item,
        is_item_use_pre_acts=args.is_item_use_pre_acts,
    )
    if args.device == "cpu":
        device = args.device
    elif args.device.startswith("cuda:"):
        device = torch.device(args.device)
        # predict_model = predict_model.to(device=device)
        predict_model.to_device(device=device)
    else:
        try:
            device_ids = list(map(int, args.device.split(',')))
            device_ids = list(range(len(device_ids)))
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            device = torch.device("cuda:0")
            predict_model.parallel_to_device(device_ids=device_ids)
        except:
            log_file.write(f"[Args Error]: Device {args.device}!\n".encode())
            log_file.flush()
            raise NotImplementedError
    model_path = os.path.join(args.model_save_path, args.model_version, f'{args.model_name}.pth')
    state = torch.load(model_path, map_location=torch.device(f"{device}"))
    if args.device.startswith("cuda:"):
        predict_model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        # predict_model = predict_model.to(device=args.device)
    else:
        predict_model.load_state_dict(state)
    predict_model.eval()
    log_file.write(f"Environment has been Constructed, Time: {st1-start}s! Load Model from {model_path} Successfully, Time {time.time()-st1}s!\n".encode())
    log_file.write(f"Initial State Vio Cost: {init_vio_cost}, {init_vio_info}\n".encode())
    log_file.flush()
    
    done = False
    total_reward, total_real_reward, migration_cost = 0, 0, 0
    move_seq, real_reward_list = [], []
    while not done:
        st = time.time()
        n_items_list = torch.tensor([graph.nitems], dtype=int).to(device)
        n_boxes_list = torch.tensor([graph.nboxes], dtype=int).to(device)
        # print(graph.edge_attr.shape)
        items_output_feats, boxes_output_feats, items_feats = predict_model.preforward(
            item_features=graph.item_features.to(device), 
            edge_indices=graph.edge_index.to(device), 
            edge_features=graph.edge_attr.to(device), 
            box_features=graph.box_features.to(device), 
            n_items_list=n_items_list, 
            n_boxes_list=n_boxes_list,
            n_pre_actions=torch.LongTensor([graph.num_pre_acts]).to(device),
            pre_actions=graph.pre_actions.to(device),
        )
        # pdb.set_trace()
        items_pred = predict_model.forward_items(items_output_feats=items_output_feats)
        
        boxes_pred = predict_model.forward_boxes(
            boxes_output_feats=boxes_output_feats, 
            items_feats=items_feats, 
            n_items_list=n_items_list, 
            n_boxes_list=n_boxes_list, 
            selected_item_idxes=items_pred.argmax(dim=-1),
        )
        
        item_idx, box_idx = 0, 0    
        for ind in range(len(n_items_list)):
            this_item_pred = items_pred[item_idx:(item_idx+n_items_list[ind])].reshape(1, -1)
            if args.item_mask:
                this_item_pred[(graph.item_masks[item_idx:(item_idx+n_items_list[ind])].to(device)<=0).reshape(1, -1)] = -1e30
            this_item_prob = torch.nn.Softmax(dim=-1)(this_item_pred).cpu().numpy().reshape(-1)
            
            this_box_pred = boxes_pred[box_idx:(box_idx+n_boxes_list[ind])].reshape(1, -1)
            # pdb.set_trace()
            if args.box_mask:
                box_masks = torch.LongTensor(action_available[this_item_prob.argmax(axis=-1)]).reshape(1, -1).to(device)
                this_box_pred[box_masks<=0] = -1e30
            this_box_prob = torch.nn.Softmax(dim=-1)(this_box_pred).cpu().numpy().reshape(-1)
            item_idx += n_items_list[ind]
            box_idx += n_boxes_list[ind]
            
            this_action = [this_item_prob.argmax(axis=-1), this_box_prob.argmax(axis=-1)]
            state, reward, done, info = ecs_env.step(this_action, is_act_dense=True)
            
            log_info = f"Step {info['move_count']}: reward {reward}, real reward {info['real_reward']}, done {done}, dense action {this_action}, {info['action_info']}, Elapsed Time {time.time()-st}s!\n"
            log_file.write(log_info.encode())
            log_file.write(f"SVio {info['move_count']}: {info['vio_info']}\n".encode())
            log_file.flush()
            
            total_reward += reward
            migration_cost += info['migration_cost']
            total_real_reward += info['real_reward']
            move_seq.append(info['real_action'])
            real_reward_list.append(info['real_reward'])
            
            if (info['real_reward'] > 500):
                is_store_pre, ecs_env_pre, pre_idx = False, None, -1
            else:
                if not is_store_pre:
                    is_store_pre, ecs_env_pre, pre_idx = True, dcp(ecs_env), info['move_count']
                    reward_pre, real_reward_pre = total_reward, total_real_reward
            
            action_available = np.asarray(state[-2].todense())
            graph = dataset.process_and_get_graph(state)
        if (time.time() - start) > args.time_limit:
            log_file.write(f"Running {args.data_path} Exceed Timelimit {args.time_limit}s!\n".encode())
            break
    
    def _undo_move(env, undo_env, action, num_undo_move, check_action_list=None):
        st1 = time.time()
        undo_flag = _check_action_can_undo(action, env, undo_env, check_action_list)
        # pdb.set_trace()
        reward, real_reward = 0, 0
        if undo_flag:
            _, reward, done, infos = env.undo_step(list([action[0], action[2], action[1]]))
            action_info, real_reward = infos["action_info"], infos['real_reward']
            # total_reward += reward
            # total_real_reward += infos['real_reward']
            st2 = time.time()
            log_file.write(f"[Undo] Step {num_undo_move}: reward {reward}, real reward {infos['real_reward']}, done {done}, {action_info}, Elapsed Time {st2-st1}s!\n".encode())
            log_file.write(f"[Undo] SVio {num_undo_move}: {infos['vio_info']}\n".encode())
            log_file.flush()
        return undo_flag, reward, real_reward
    if args.is_undo_move:
        if pre_idx >= 0:
            ecs_env = ecs_env_pre
            move_seq, real_reward_list = move_seq[:pre_idx], real_reward_list[:pre_idx]
            total_reward, total_real_reward = reward_pre, real_reward_pre
            log_file.write(f"Set Env to Step {pre_idx}, and Start Undoing...\n".encode())
            log_file.flush()
        num_undo_move = 0
        move_seq = np.array(move_seq[::-1])
        undo_flags = np.ones(len(move_seq), dtype=bool)
        undo_env = dcp(ecs_env)
        to_boxes_map = {i: [] for i in range(ecs_env.box_count)}
        for i in range(len(move_seq)):
            undo_env.undo_step([move_seq[i][0], move_seq[i][2], move_seq[i][1]])
            undo = False
            if real_reward_list[-i-1] <= 500:
                undo, r, rr = _undo_move(ecs_env, undo_env, move_seq[i], num_undo_move, list(move_seq[to_boxes_map[move_seq[i][1]]])[::-1])
                if undo:
                    undo_flags[i] = False
                    total_reward += r
                    total_real_reward += rr
                    num_undo_move += 1
            if not undo:
                to_boxes_map[move_seq[i][2]].append(i)
                    
        move_seq = move_seq[undo_flags][::-1]
    final_score = ecs_env.get_cur_state_score()
    final_vio_cost, final_vio_info = ecs_env.get_cur_state_vio_cost()
    end1 = time.time()
    log_file.write(f"Init Score: {init_score}, Final Score: {final_score}, Num Action: {len(move_seq)}, Migration Cost: {migration_cost}, Total Reward: {total_reward}, Total Real Reward: {total_real_reward}! Elapsed Time: {end1-start}s!\n".encode())
    log_file.write(f"Final State Vio Cost: {final_vio_cost}, {final_vio_info}\n".encode())
    log_file.flush()
    
    if not args.is_limited_count:
        action_list = np.array(move_seq)[:, [0,2]].tolist()
        ecs_env = ECSRawDenseEnvironment(data_path=args.data_path, is_limited_count=True,
                                         is_filter_unmovable=args.is_filter_unmovable, is_dense_items=args.is_dense_items,
                                         is_dense_boxes=args.is_dense_boxes, is_state_merge_placement=args.is_state_merge_placement, is_process_numa=args.is_process_numa)
        total_reward, total_real_reward, num_action, invalid_num = ecs_env.eval_move_sequence(action_list, is_act_dense=False)
        final_score = ecs_env.get_cur_state_score()
        log_file.write(f"Init Score: {init_score}, Final Score: {final_score}, Num Action: {num_action}, Total Reward: {total_reward}, Total Real Reward: {total_real_reward}! Elapsed Time: {end1-start}s! Total Time: {time.time()-start}s!\n".encode())
        

if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--is_use_full_infos", action='store_true', default=False)
    parser.add_argument("--emb_size_list", nargs='+', type=int, default=[64, 32])
    parser.add_argument("--out_emb_size_list", nargs='+', type=int, default=[32])
    parser.add_argument("--n_gnn_layers", type=int, default=2)
    parser.add_argument("--is_gnn_resnet", action='store_true', default=False)
    parser.add_argument("--graph_type", type=str, default='gcn', choices=['gcn', 'gin', 'gat'])
    parser.add_argument("--normalization", type=str, default='batch', choices=['batch', 'layer', 'none'])
    parser.add_argument("--activate_func", type=str, default='ReLU')
    parser.add_argument("--is_box_outfeats_use_item", action='store_false', default=True)
    parser.add_argument("--is_item_use_pre_acts", action='store_true', default=False)
    parser.add_argument("--device", type=str, default="2,3")
    parser.add_argument("--item_mask", action='store_false', default=True)
    parser.add_argument("--box_mask", action='store_false', default=True)
    
    parser.add_argument("--is_limited_count", action='store_false', default=True)
    parser.add_argument("--is_filter_unmovable", action='store_false', default=True)
    parser.add_argument("--is_dense_items", action='store_false', default=True)
    parser.add_argument("--is_dense_boxes", action='store_true', default=False)
    parser.add_argument("--is_state_merge_placement", action='store_true', default=False)
    parser.add_argument("--is_process_numa", action="store_false", default=True)
    parser.add_argument("--is_undo_move", action='store_true', default=False)
    parser.add_argument("--time_limit", type=int, default=1200)
    
    parser.add_argument("--data_path", type=str, default="../data/ecs_data")
    parser.add_argument("--model_version", type=str, default="ecs_gnn_v0_t-1")
    parser.add_argument("--train_set_id", type=int, default=-1)
    parser.add_argument("--model_save_path", type=str, default="../results/ecs_imitation/results/pretrain")
    parser.add_argument("--model_name", type=str, default="model_best")
    parser.add_argument("--log_save_path", type=str, default="../results/ecs_imitation/results/eval_logs")
    parser.add_argument("--is_eval_all", action='store_true', default=False)

    args = parser.parse_args()
    
    data_root_path = args.data_path
    log_save_path = f'{args.log_save_path}/{args.model_version}_{args.model_name}'
    if not args.is_limited_count:
        log_save_path = f'{log_save_path}_unlimited'
    if args.is_undo_move:
        log_save_path = f'{log_save_path}_undo_move'
    os.makedirs(log_save_path, exist_ok=True)
    def _task(data):
        args.data_path = os.path.join(data_root_path, data)
        args.log_save_path = f"{log_save_path}/{args.data_path.split('/')[-1]}_{args.model_name}.log"
        if not os.path.exists(args.log_save_path):
            torch.random.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            random.seed(0)
            
            eval_one_epoch(args=args)
    
    with open('../data/ecs_data_classification.json', 'r') as f:
        data_classified = json.load(f)
        data_list = [data_classified[f"type_{i}"] for i in range(len(data_classified))]
        for i in range(len(data_list)):
            for j in range(len(data_list[i])):
                data_list[i][j] = str(data_list[i][j])

    if (args.train_set_id < 0) or (args.train_set_id >= len(data_list)) or args.is_eval_all:
        eval_data_list = data_list
    else:
        eval_data_list = [data_list[args.train_set_id]]

    for datas in eval_data_list:
        for data in datas:
            _task(str(data))

