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
from ecs_imitation.plot_figures import plot_train_results
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
from scipy import sparse
from ecs_env.ecs_env_binary_wrapper import ECSRawDenseEnvironment
import pdb
from copy import deepcopy as dcp
import json


global data_full_list
global num_datas
with open('../data/ecs_data_classification.json', 'r') as f:
    data_classified = json.load(f)
    data_full_list = [data_classified[f"type_{i}"] for i in range(len(data_classified))]
    num_datas = 0
    for i in range(len(data_full_list)):
        num_datas += len(data_full_list[i])
        for j in range(len(data_full_list[i])):
            data_full_list[i][j] = str(data_full_list[i][j])


"""is_item_use_pre_acts, parallel evalution"""
def train_one_epoch(predict_model, data_loader, args, optimizer=None, device='cpu'):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    if optimizer:
        predict_model.train()
    else:
        predict_model.eval()
    mean_item_loss, mean_box_loss = 0, 0
    mean_item_acc, mean_box_acc, mean_acc = 0, 0, 0
    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, batch in enumerate(data_loader):
            batch = batch.to(device)

            n_items_list, n_boxes_list = batch.nitems, batch.nboxes
            selected_item_idxes = batch.actions[:, 0]
            items_pred, boxes_pred, items_output_feats, boxes_output_feats = predict_model(
                item_features=batch.item_features, 
                edge_indices=batch.edge_index, 
                edge_features=batch.edge_attr, 
                box_features=batch.box_features, 
                selected_item_idxes=selected_item_idxes, 
                n_items_list=n_items_list, 
                n_boxes_list=n_boxes_list,
                n_pre_actions=batch.num_pre_acts,
                pre_actions=batch.pre_actions,
            )
    
            # compute loss
            item_loss, box_loss = 0, 0
            item_acc, box_acc, acc = 0, 0, 0
            item_idx, box_idx = 0, 0
            
            for ind in range(len(n_items_list)):
                this_item_pred = items_pred[item_idx:(item_idx+n_items_list[ind])].reshape(1, -1)
                this_item_feats = batch.item_features[item_idx:(item_idx+n_items_list[ind])]
                if args.item_mask:
                    this_item_pred[(batch.item_masks[item_idx:(item_idx+n_items_list[ind])]<=0).reshape(1, -1)] = -1e30
                this_box_pred = boxes_pred[box_idx:(box_idx+n_boxes_list[ind])].reshape(1, -1)
                this_box_feats = batch.box_features[box_idx:(box_idx+n_boxes_list[ind])]
                if args.box_mask:
                    this_box_pred[(batch.box_masks[box_idx:(box_idx+n_boxes_list[ind])]<=0).reshape(1, -1)] = -1e30
                item_idx += n_items_list[ind]
                box_idx += n_boxes_list[ind]
                
                item_sol = batch.actions[ind, :1].reshape(-1)
                box_sol = batch.actions[ind, 1:].reshape(-1)
                # pdb.set_trace()
                this_item_loss = torch.nn.CrossEntropyLoss()(this_item_pred, item_sol)
                this_box_loss = torch.nn.CrossEntropyLoss()(this_box_pred, box_sol)
                item_loss += this_item_loss
                box_loss += this_box_loss
                
                item_pred_idx, item_sol_idx = this_item_pred[0].argmax().item(), item_sol[0].item()
                ic = int(this_item_feats[item_pred_idx].equal(this_item_feats[item_sol_idx]))
                box_pred_idx, box_sol_idx = this_box_pred[0].argmax().item(), box_sol[0].item()
                bc = int(this_box_feats[box_pred_idx].equal(this_box_feats[box_sol_idx]))

                item_acc += ic
                box_acc += bc
                acc += int((ic + bc) == 2)
                # pdb.set_trace()
            
            if optimizer is not None:
                optimizer.zero_grad()
                (item_loss + box_loss).backward()
                optimizer.step()
            
            mean_item_loss += item_loss.item()
            mean_box_loss += box_loss.item()
            mean_item_acc += item_acc
            mean_box_acc += box_acc
            mean_acc += acc
            n_samples_processed += batch.num_graphs
            print(f"{step}, Item loss: {item_loss.item():0.3f}, Box loss: {box_loss.item():0.3f}, Item acc: {item_acc/batch.num_graphs:0.3f}, Box acc: {box_acc/batch.num_graphs:0.3f}, acc: {acc/batch.num_graphs:0.3f}.")
            # torch.cuda.empty_cache()
    
    mean_item_loss /= n_samples_processed
    mean_box_loss /= n_samples_processed
    mean_item_acc /= n_samples_processed
    mean_box_acc /= n_samples_processed
    mean_acc /= n_samples_processed

    return mean_item_loss, mean_box_loss, mean_item_acc, mean_box_acc, mean_acc


@torch.no_grad()
def eval_one_epoch(data_path, device, model_path, args):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    # print(data_path)
    start = time.time()
    dataset = GraphDataset([data_path])
    
    ecs_env = ECSRawDenseEnvironment(data_path=data_path, is_limited_count=args.is_limited_count,
                                     is_filter_unmovable=args.is_filter_unmovable, is_dense_items=args.is_dense_items,
                                     is_dense_boxes=args.is_dense_boxes, is_state_merge_placement=args.is_state_merge_placement,
                                     is_process_numa=args.is_process_numa)
    state = ecs_env.reset()
    action_available = np.asarray(state[-2].todense())
    if action_available.sum() <= 0:
        return ecs_env.get_cur_state_score(), 0
    graph = dataset.process_and_get_graph(state)
    
    # device = f'cuda:{device}'
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
    predict_model.parallel_to_device(device_ids=[device])
    state = torch.load(model_path, map_location=torch.device(f"cuda:{device}"))
    predict_model.load_state_dict(state)
    predict_model.eval()
    
    done = False
    total_reward, total_real_reward = 0, 0
    while not done:
        n_items_list=torch.tensor([graph.nitems], dtype=int).to(device)
        n_boxes_list=torch.tensor([graph.nboxes], dtype=int).to(device)
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
            
            total_reward += reward
            total_real_reward += info['real_reward']
            
            action_available = np.asarray(state[-2].todense())
            graph = dataset.process_and_get_graph(state)
        if (time.time() - start) > args.time_limit:
            break
            
    final_score = ecs_env.get_cur_state_score()
    return final_score, total_real_reward


def train_and_eval_model(train_files, valid_files, train_names, valid_names,
                         log_save_path, model_save_path, result_save_path, figure_save_path, 
                         args):
    log_file = open(log_save_path, 'wb')
    log_file.write((str(train_names) + '\n').encode())
    log_file.write((str(valid_names) + '\n').encode())
    log_file.flush()
    
    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.loader.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loaders = [torch_geometric.loader.DataLoader(
        GraphDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0)
        for val_data in valid_files
    ]
    
    first_graph = train_data.get(0)
    log_file.write(f"Item Feats: {first_graph.n_item_feats}, Box Feats: {first_graph.n_box_feats}, Edge Feats: {first_graph.n_edge_feats}!\n".encode())
    log_file.flush()
    PredictModel = GNNPolicy(
        n_item_feats=first_graph.n_item_feats, 
        n_box_feats=first_graph.n_box_feats, 
        n_edge_feats=first_graph.n_edge_feats,
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
    args.n_item_feats, args.n_box_feats, args.n_edge_feats = first_graph.n_item_feats, first_graph.n_box_feats, first_graph.n_edge_feats
    if args.device == "cpu":
        device = args.device
    elif args.device.startswith("cuda:"):
        device = args.device
        PredictModel = PredictModel.to(device)
    else:
        try:
            device_ids = list(map(int, args.device.split(',')))
            device_ids = list(range(len(device_ids)))
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device
            device = torch.device("cuda:0")
            PredictModel.parallel_to_device(device_ids=device_ids)
        except:
            log_file.write(f"[Args Error]: Device {args.device}!\n".encode())
            log_file.flush()
            raise NotImplementedError

    optimizer = torch.optim.Adam(PredictModel.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_acc, best_epoch = -1, -1
    result_history = {
        'train_item_loss': [], 'train_box_loss': [], 'train_item_acc': [], 'train_box_acc': [], 'train_acc': [], 'train_final_score': [], 'train_reward': [],
        'valid_item_loss': [], 'valid_box_loss': [], 'valid_item_acc': [], 'valid_box_acc': [], 'valid_acc': [], 'valid_final_score': [], 'valid_reward': [],
    }
    log_file.write(f"Train and Valid Start...\n".encode())
    log_file.flush()
    valid_epochs = []
    for epoch in range(args.epochs):
        begin = time.time()
        # args.i_epoch = epoch
        train_item_loss, train_box_loss, train_item_acc, train_box_acc, train_acc = train_one_epoch(
            predict_model=PredictModel, 
            data_loader=train_loader, 
            args=args, 
            optimizer=optimizer, 
            device=device
        )
        st = f'@Epoch{epoch}, Train Item loss:{train_item_loss}, Train Box loss: {train_box_loss}, Train Item acc: {train_item_acc}, Train Box acc: {train_box_acc}, Train acc: {train_acc}!\n'
        torch.save(PredictModel.state_dict(), model_save_path+'model_last.pth')
        if (epoch % args.test_iterval == 0) or (epoch == args.epochs-1):
            valid_epochs.append(epoch)
            train_final_score, train_reward = np.zeros(len(train_names)), np.zeros(len(train_names))
            # for ti, name in enumerate(train_names):
            #     tr_fs, tr_r = eval_one_epoch(
            #         data_path=os.path.join(args.data_path, name),
            #         predict_model=PredictModel,
            #         device=device,
            #         args=args,
            #     )
            #     train_final_score[ti] = tr_fs
            #     train_reward[ti] = tr_r
            valid_item_loss, valid_box_loss, valid_item_acc, valid_box_acc, valid_acc, valid_final_score, valid_reward = [], [], [], [], [], np.zeros(len(valid_names)), np.zeros(len(valid_names))
            for vi, (valid_loader, name) in enumerate(zip(valid_loaders, valid_names)):
                vil, vbl, via, vba, va = train_one_epoch(
                    predict_model=PredictModel, 
                    data_loader=valid_loader, 
                    args=args, 
                    optimizer=None, 
                    device=device
                )
                valid_item_loss.append(vil)
                valid_box_loss.append(vbl)
                valid_item_acc.append(via)
                valid_box_acc.append(vba)
                valid_acc.append(va)
            
            for vi, name in enumerate(valid_names):
                va_fs, va_r = eval_one_epoch(os.path.join(args.data_path, name), 3, model_save_path+'model_last.pth', args)
                valid_final_score[vi] = va_fs
                valid_reward[vi] = va_r
                
            st += f'@Epoch{epoch}, Train Final Score: {np.mean(train_final_score)}[{np.min(train_final_score)}, {np.max(train_final_score)}], Train Reward: {np.mean(train_reward)}[{np.min(train_reward)}, {np.max(train_reward)}]!\n'
            st += f'@Epoch{epoch}, Valid Item loss:{np.mean(valid_item_loss)}, Valid Box loss: {np.mean(valid_box_loss)}, Valid Item acc: {np.mean(valid_item_acc)}[{np.min(valid_item_acc)}, {np.max(valid_item_acc)}], Valid Box acc: {np.mean(valid_box_acc)}[{np.min(valid_box_acc)}, {np.max(valid_box_acc)}], Valid acc: {np.mean(valid_acc)}[{np.min(valid_acc)}, {np.max(valid_acc)}]!\n'
            st += f'@Epoch{epoch}, Valid Final Score: {np.mean(valid_final_score)}[{np.min(valid_final_score)}, {np.max(valid_final_score)}], Valid Reward: {np.mean(valid_reward)}[{np.min(valid_reward)}, {np.max(valid_reward)}]!\n'
            st += f'@Epoch{epoch}, {valid_final_score}\n'
            st += f'@Epoch{epoch}, {valid_reward}\n'
            if np.mean(valid_reward) > best_val_acc:
                best_val_acc, best_epoch = np.mean(valid_reward), epoch
                torch.save(PredictModel.state_dict(), model_save_path+'model_best.pth')
                st += f"[Store Best Valid Acc Model] Epoch {best_epoch}, Train Item acc {train_item_acc}, Valid Item acc {np.mean(valid_item_acc)} Valid Reward {np.mean(valid_reward)}!\n"
                # log_file.write(st.encode())
                # log_file.flush()
            
        if (epoch % args.save_interval == 0):
            torch.save(PredictModel.state_dict(), model_save_path+'model_{}.pth'.format(epoch))
        
        st += f'@Epoch{epoch}, Spend Time: {time.time()-begin}s...\n'
        log_file.write(st.encode())
        log_file.flush()
    
        for key in result_history.keys():
            if key in locals().keys():
                result_history[key].append(locals()[key])
    
    np.save(result_save_path, result_history)

    plot_train_results(result_history=result_history, figure_save_path=figure_save_path)
    
    log_file.write(f"Train and Valid Done!\n".encode())
    log_file.flush()


def get_train_and_valid_files(args):
    train_files, valid_files = [], []
    for data_type in args.data_type_list:
        this_path = os.path.join(args.expert_data_path, data_type)
        sample_names = sorted(os.listdir(this_path))
        vstart_idx, vprop = args.valid_start_idx, args.valid_prop
        if (vprop > 0) and (vprop < 0.5):
            vgap = int(np.ceil(1/vprop))
            valid_names = (sample_names[vstart_idx:] + sample_names[:vstart_idx])[::vgap]
            train_names = list(set(sample_names) - set(valid_names))
        elif (vprop >= 0.5) and (vprop < 1):
            vgap = int(np.ceil(1/(1-vprop)))
            train_names = (sample_names[vstart_idx:] + sample_names[:vstart_idx])[::vgap]
            # valid_names = list(set(sample_names) - set(train_names))
            valid_names = (sample_names[vstart_idx+3:] + sample_names[:(vstart_idx+3)])[::(vgap*3)]
        else:
            if args.train_set_id >= 0 and args.train_set_id < len(data_full_list):
                this_set = data_full_list[args.train_set_id]
                this_set = this_set[:int(len(this_set) * 0.6)]
                train_gap = int(np.ceil(len(this_set) / max(min(int(len(this_set)*0.2), 50), 1)))
                train_names = this_set[::train_gap]
                valid_gap = int(np.ceil(len(this_set) / max(min(int(len(this_set)*0.1), 5), 1)))
                valid_names = (this_set[1:] + this_set[:1])[::valid_gap]
                # valid_names = [valid_names[0], valid_names[-1]] if len(valid_names) > 2 else valid_names
            else:
                n_train_samples, n_valid_samples = 200, 5
                train_names, valid_names = [], []
                for i in range(len(data_full_list)):
                    n_samples = int(n_train_samples * np.clip(len(data_full_list[i]) / num_datas, 0.1, 0.7))
                    train_gap = int(np.ceil(len(data_full_list[i]) / max(n_samples, 5)))
                    train_names.extend(data_full_list[i][::train_gap])
                    n_samples = int(n_valid_samples * len(data_full_list[i]) / num_datas)
                    valid_gap = int(np.ceil(len(data_full_list[i]) / max(n_samples, 2)))
                    valid_names.extend((data_full_list[i][10:] + data_full_list[i][:10])[::valid_gap])
                
        for name in train_names:
            if not os.path.exists(os.path.join(this_path, name)):
                continue
            # print(len(os.listdir(os.path.join(this_path, name))))
            for file in os.listdir(os.path.join(this_path, name)):
                if file != 'edge_feats.pkl':
                    train_files.append((os.path.join(this_path, name, file), os.path.join(this_path, name, 'edge_feats.pkl')))
        
        for name in valid_names:
            if not os.path.exists(os.path.join(this_path, name)):
                continue
            this_valid = []
            for file in os.listdir(os.path.join(this_path, name)):
                if file != 'edge_feats.pkl':
                    this_valid.append((os.path.join(this_path, name, file), os.path.join(this_path, name, 'edge_feats.pkl')))
            valid_files.append(this_valid)
    
    random.shuffle(train_files)
    print("Num TrainFiles:", len(train_files))
    print("Num ValidFiles:", len(valid_files))
    if not os.path.isdir(f'{args.save_path}/train_logs/{args.model_version}'):
        os.makedirs(f'{args.save_path}/train_logs/{args.model_version}')
    if not os.path.isdir(f'{args.save_path}/pretrain/{args.model_version}'):
        os.makedirs(f'{args.save_path}/pretrain/{args.model_version}')
    if not os.path.isdir(f'{args.save_path}/result_history'):
        os.makedirs(f'{args.save_path}/result_history')
    if not os.path.isdir(f'{args.save_path}/result_figures'):
        os.makedirs(f'{args.save_path}/result_figures')
    model_save_path = f'{args.save_path}/pretrain/{args.model_version}/'
    with open(os.path.join(model_save_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f, default_flow_style=False)
    log_save_path = f'{args.save_path}/train_logs/{args.model_version}/{args.model_version}_train.log'
    result_save_path = f'{args.save_path}/result_history/{args.model_version}.npy'
    figure_save_path = f'{args.save_path}/result_figures/{args.model_version}'
    
    return train_files, valid_files, train_names, valid_names, model_save_path, log_save_path, result_save_path, figure_save_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    
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
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--item_mask", action='store_false', default=True)
    parser.add_argument("--box_mask", action='store_false', default=True)
    
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5)
    
    parser.add_argument("--is_limited_count", action='store_false', default=True)
    parser.add_argument("--is_filter_unmovable", action='store_false', default=True)
    parser.add_argument("--is_dense_items", action='store_false', default=True)
    parser.add_argument("--is_dense_boxes", action='store_true', default=False)
    parser.add_argument("--is_state_merge_placement", action='store_true', default=False)
    parser.add_argument("--is_process_numa", action="store_false", default=True)
    parser.add_argument("--time_limit", type=int, default=1200)
    
    parser.add_argument("--expert_data_path", type=str, default="../results/ecs_imitation/expert_data_merge_placement")
    parser.add_argument("--data_type_list", nargs='+', type=str, default=['ecs_data'])
    parser.add_argument("--valid_start_idx", type=int, default=0)
    parser.add_argument("--valid_prop", type=float, default=-1)
    parser.add_argument("--train_set_id", type=int, default=-1)
    parser.add_argument("--test_iterval", type=int, default=10)
    parser.add_argument("--model_version", type=str, default="ecs_gnn_v0_t-1")
    parser.add_argument("--save_path", type=str, default="../results/ecs_imitation/results")
    parser.add_argument("--data_path", type=str, default="../data/ecs_data")

    args = parser.parse_args()
    
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    
    train_files, valid_files, train_names, valid_names, model_save_path, log_save_path, result_save_path, figure_save_path = get_train_and_valid_files(args)
    
    train_and_eval_model(
        train_files=train_files,
        valid_files=valid_files,
        train_names=train_names,
        valid_names=valid_names,
        log_save_path=log_save_path,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        figure_save_path=figure_save_path,
        args=args,
    )

