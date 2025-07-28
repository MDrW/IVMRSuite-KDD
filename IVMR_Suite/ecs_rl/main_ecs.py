# Copyright (C) 2022. ByteDance Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the Apache-2.0 license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache-2.0 License for more details.


"""
VM: cpu, cpu, mem, mem, cpu % 16, cpu % 16  (0 is full, 1 is empty)
PM: cpu, cpu, mem, mem, fragment_rate, cpu % 16, fragment_rate, cpu % 16
cpu % 16 = round(normalized_cpu * 88) % 16 / 16
fragment_rate = round(normalized_cpu * 88) % 16 / round(normalized_cpu * 88)
To rescale memory, mem * 368776
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import pandas as pd
import wandb

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attn", help="model architecture")
    # ecs v1
    parser.add_argument("--data-path", type=str, default="821636", help="ecs data path")
    parser.add_argument("--eval-data-path", type=str, default="821636", help="ecs data path")
    # ecs v2
    parser.add_argument("--base-path", type=str, default="/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy", help="ecs data path")
    parser.add_argument("--data-list", type=list, default=['8616'], help="ecs data path")
    parser.add_argument("--eval-data-list", type=list, default=['8616'], help="ecs data path")
    parser.add_argument("--type", type=str, default='type_1', help="ecs data type")
    
    parser.add_argument("--pretrain", action='store_true',
                        help="if toggled, we will restore pretrained weights for vm selection")
    parser.add_argument("--gym-id", type=str, default="ecs-v2", # generalizer-v1
                        help="the id of the gym environment")
    parser.add_argument("--vm-data-size", type=str, default="M", choices=["M", "L"],
                        help="size of the dataset")
    parser.add_argument("--max-steps", type=int, default=50, help="maximum number of redeploy steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=4000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--normalize", action='store_true',
                        help="if toggled, we will normalize the input features")
    parser.add_argument("--track", action='store_true',
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--debug", action='store_true',
                        help="if toggled, this experiment will save run details")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--accum-iter", type=int, default=4,
                        help="number of iterations where gradient is accumulated before the weights are updated;"
                             " used to increase the effective batch size")
    parser.add_argument("--update-epochs", type=int, default=2,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.005,  # 0.01
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1e-4,  # 1e-4
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) 
    # args.batch_size = 128
    args.minibatch_size = int(args.batch_size // (args.num_minibatches * args.accum_iter))
    return args


def make_env(gym_id, seed, data, mode='train'):
    def thunk():
        if gym_id == 'ecs-v1':
            env = gym.make(gym_id, data_path=data)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            return env
        elif gym_id == 'ecs-v2':
            env = gym.make(gym_id, data_list=data, mode=mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            return env
    return thunk


class CategoricalMasked(Categorical):
    def __init__(self, logits=None, probs=None, masks=None):
        if masks is None or torch.sum(masks) == 0:
            self.masks = None
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.masks = masks
            if logits is not None:
                logits = torch.where(self.masks, torch.tensor(-1e8, device=logits.device), logits)
                super(CategoricalMasked, self).__init__(logits=logits)
            else:
                probs = torch.where(self.masks, torch.tensor(0.0, device=probs.device), probs)
                small_val_mask = torch.sum(probs, dim=1) < 1e-4
                probs[small_val_mask] = torch.where(self.masks[small_val_mask], torch.tensor(0.0, device=probs.device),
                                                    torch.tensor(1.0, device=probs.device))
                super(CategoricalMasked, self).__init__(probs=probs)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, torch.tensor(0.0, device=p_log_p.device), p_log_p)
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, envs, vm_net, pm_net, params, args_model):
        super(Agent, self).__init__()

        self.vm_net = vm_net
        self.pm_net = pm_net
        self.nvec = np.array([envs.single_action_space.nvec])
        self.device = params.device
        self.num_vm = envs.num_items
        self.model = args_model
        self.envs = envs
        
    def get_value(self, obs_info_pm, obs_info_all_vm, obs_info_num_steps, obs_info_num_vms):
        vm_mask = torch.tensor(np.array(self.envs.call_parse('get_vm_dense_mask', id=[1] * args.num_envs)), dtype=torch.bool, device=self.device)
        # num_vms_mask = torch.arange(self.num_vm, device=obs_info_all_vm.device)[None, :] >= obs_info_num_vms[:, None]
        if "attn" in self.model:
            return self.vm_net(obs_info_all_vm, obs_info_num_steps, obs_info_pm, vm_mask)[1]
        elif self.model == "mlp":
            return self.vm_net(obs_info_all_vm, obs_info_pm)[1]

    def get_action_and_value(self, envs, obs_info_pm, obs_info_all_vm, obs_info_num_steps, obs_info_num_vms,
                             vm_mask=None, pm_mask=None, selected_vm=None, selected_pm=None):
        if pm_mask is None:
            assert selected_vm is None and selected_pm is None, \
                'action must be None when action_mask is not given!'
        else:
            assert selected_vm is not None and selected_pm is not None, \
                'action must be given when action_mask is given!'
        # temp = torch.arange(self.num_vm, device=self.device)[None, :] >= obs_info_num_vms[:, None]
        # print("temp", temp.shape)self.vm_net
        if vm_mask is None:
            vm_mask = torch.tensor(np.array(envs.call_parse('get_vm_dense_mask', id=[1] * args.num_envs)), dtype=torch.bool, device=self.device)  # pm_mask:  torch.Size([8, 279])
        # print("num_vms_mask", num_vms_mask.shape)
        b_sz = obs_info_pm.shape[0]
        # obs_info_all_vm: torch.Size([8, num_vm, 9])
        # obs_info_pm:  torch.Size([8, num_pm, 7])


        if self.model == "attn":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_num_steps, obs_info_pm, vm_mask)
        elif self.model == "mlp":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_pm)
        else:
            raise ValueError(f'self.model={self.model} is not implemented')

        # vm_pred:  torch.Size([8, num_vm])
        # critic_score:  torch.Size([8, 1])
        vm_cat = CategoricalMasked(logits=vm_logits, masks=vm_mask)
        if selected_vm is None:
            selected_vm = vm_cat.sample()
        vm_log_prob = vm_cat.log_prob(selected_vm)
        # selected_vm:  torch.Size([8])
        # vm_log_prob:  torch.Size([8])
        # entropy:  torch.Size([8])
        # print("selected_vm", selected_vm)
        # print("max", torch.max(vm_cat.probs, dim=1))
        # print("current", vm_cat.probs[0][selected_vm])
        # print("vm_mask", vm_mask[0][selected_vm])
        if pm_mask is None:
            pm_mask = torch.tensor(np.array(envs.call_parse('get_pm_mask', item_id=selected_vm.cpu().tolist())),
                                   dtype=torch.bool, device=self.device)  # pm_mask:  torch.Size([8, 279])

        # obs_info_all_vm:  torch.Size([8, 2089, 14])
        if "attn" in self.model:
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_num_steps,
                                    obs_info_pm)  # b_sz
        elif self.model == "mlp":
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_pm)  # b_sz
        else:
            raise ValueError(f'self.model={self.model} is not implemented')
        # pm_logits:  torch.Size([8, 279])
        pm_cat = CategoricalMasked(logits=pm_logits, masks=pm_mask)
        if selected_pm is None:
            selected_pm = pm_cat.sample()  # selected_pm:  torch.Size([8])
        pm_log_prob = pm_cat.log_prob(selected_pm)  # pm_log_prob:  torch.Size([8])

        return selected_vm, vm_log_prob, vm_cat.entropy(), selected_pm, pm_log_prob, \
               pm_cat.entropy(), critic_score, pm_mask, vm_mask



# type_datasets_dict = {
#     "type_1": ['10003', '10027', '10056', '10065', '10104', '10112', '10135', '10140', '10187', '10506', '10752', '10784', '10809', '10839', '10875', '10908', '10931', '11349', '11368', '11509', '11778', '11833', '12027', '12067', '12553', '12574', '13010', '13317', '13650', '13675', '13691', '13710', '13727', '13741', '14061', '14133', '14172', '14206', '14236', '14328', '14370', '14409', '14494', '14614', '14633', '14642', '821464', '8621', '8674', '8706', '8780', '8828', '8832', '8855', '8975', '8994', '9075'],
#     # "type_2": ['10002', '10016', '10022', '10023', '10031', '10036', '10057', '10058', '10062', '10086', '10095', '10114', '10124', '10156', '10158', '10167', '10177', '10221', '10514', '10726', '10778', '10813', '10848', '10881', '10898', '10954', '11344', '11367', '11529', '11834', '12055', '12541', '12571', '13006', '13324', '13651', '13658', '13659', '13665', '13680', '14045', '820624', '8616', '8660', '8700', '8767', '8858', '8971', '9002', '9094', '9938'],
#     "type_2": ["10221", "14045", "13665", "10177", "10167", "11367", "10095", "11344", "13680", "10058", "12055", "10057", "12571", "820624", "11834", "13651", "12541", "13659", "13658", "9094", "10062", "8971", "9002", "10086", "10898", "11529"],
#     "type_3": ['821576', '824526', '828295', '831438', '840008', '843693', '847814', '849322', '849342', '858279', '858285', '858303', '858408', '858493', '858534', '858588'],
#     "type_4": ['841935', '842308', '842479', '842697', '843112', '843213', '844254', '844308', '844320', '844357', '845479', '845528', '845633', '845983', '846083', '846175', '846340', '846412', '846513', '846525', '846621', '846803', '847043', '847216', '847275', '847393', '847453', '847462', '847476', '856897'],
#     "type_5": ['819257', '821575', '824520', '828209', '831327', '834875', '838811', '842511', '845746', '849621', '853064'],
#     "type_6": ['817879', '820978', '824541', '828390', '832572', '836830', '841291', '845764', '849830', '854485', '858885'],
#     "type_7": ['817078', '820112', '823603', '839907', '843702', '852760', '852815', '852889', '852927', '853141'],
#     "type_8": ['823534', '824441', '825425', '826348', '827383', '828298', '829108', '829861', '831332', '832479', '833517', '834786', '836609', '837657', '838812', '839913', '841180', '842516', '843509', '844531', '845602', '846758', '847708', '857621', '858740'],
#     "type_9": ['817807', '820102', '823530', '826414', '829863', '833586', '837759', '841265', '845695', '849721', '853124', '857687'],
#     "type_10": ['818433', '818453', '818469', '818472', '818474', '818476', '818479', '818483', '818582', '818914', '824513', '828309'],
#     }

type_datasets_dict = {
    "type_1": ['10003', '10027', '10056', '10065', '10104', '10112', '10135', '10140', '10187', '10506', '10752', '10784', '10809', '10839', '10875', '10908', '10931', '11349', '11368', '11509', '11778', '11833', '12027', '12067', '12553', '12574', '13010', '13317', '13650', '13675', '13691', '13710', '13727', '13741', '14061', '14133', '14172', '14206', '14236', '14328', '14370', '14409', '14494', '14614', '14633', '14642', '821464', '8621', '8674', '8706', '8780', '8828', '8832', '8855', '8975', '8994', '9075'],
    # "type_2": ['10002', '10016', '10022', '10023', '10031', '10036', '10057', '10058', '10062', '10086', '10095', '10114', '10124', '10156', '10158', '10167', '10177', '10221', '10514', '10726', '10778', '10813', '10848', '10881', '10898', '10954', '11344', '11367', '11529', '11834', '12055', '12541', '12571', '13006', '13324', '13651', '13658', '13659', '13665', '13680', '14045', '820624', '8616', '8660', '8700', '8767', '8858', '8971', '9002', '9094', '9938'],
    # "type_2": ["10221", "14045", "13665", "10177", "10167", "11367", "10095", "11344", "13680", "10058", "12055", "10057", "12571", "820624", "11834", "13651", "12541", "13659", "13658", "9094", "10062", "8971", "9002", "10086", "10898", "11529"],
    "type_2": [
        "826417", 
        "839000", 
        "843690", 
        "847811", 
        "851838"
    ],
    "type_3": ['821576', '824526', '828295', '831438', '840008', '843693', '847814', '849322', '849342', '858279', '858285', '858303', '858408', '858493', '858534', '858588'],
    "type_4": ['841935', '842308', '842479', '842697', '843112', '843213', '844254', '844308', '844320', '844357', '845479', '845528', '845633', '845983', '846083', '846175', '846340', '846412', '846513', '846525', '846621', '846803', '847043', '847216', '847275', '847393', '847453', '847462', '847476', '856897'],
    # "type_5": ['819257', '821575', '824520', '828209', '831327', '834875', '838811', '842511', '845746', '849621', '853064'],
    # "type_5": ["817802", "820098", "823527", "826474", "829854", "833583"], # added in 11/18
    "type_5": [
        "832472", 
        "840006", 
        "843701", 
        "847735"
    ],
    # "type_6": ['817879', '820978', '824541', '828390', '832572', '836830', '841291', '845764', '849830', '854485', '858885'],
    "type_6": [
        "819402",
        "826440",
        "829957",
        "834889",
        "839017",
        "847828"
    ],
    # "type_7": ['817078', '820112', '823603', '839907', '843702', '852760', '852815', '852889', '852927', '853141'],
    "type_7": [
        "819257",
        "821575",
        "824520",
        "828209",
        "831327",
        "834875",
        "838811",
        "842511",
        "845746",
        "849621",
        "853064"
    ],
    # "type_8": ['823534', '824441', '825425', '826348', '827383', '828298', '829108', '829861', '831332', '832479', '833517', '834786', '836609', '837657', '838812', '839913', '841180', '842516', '843509', '844531', '845602', '846758', '847708', '857621', '858740'],
    "type_8": [
        "819305",
        "821629",
        "825420",
        "828357",
        "840011",
        "843677",
        "847754"
    ],
    "type_9": ['817807', '820102', '823530', '826414', '829863', '833586', '837759', '841265', '845695', '849721', '853124', '857687'],
    "type_10": ['818433', '818453', '818469', '818472', '818474', '818476', '818479', '818483', '818582', '818914', '824513', '828309'],
    "type_11": [
        "820107",
        "823440",
        "826457",
        "829767"
    ],
    "type_12": [
        "818645",
        "821521",
        "824437",
        "836605",
        "840003",
        "843678",
        "847675"
    ],
    "type_13": [
        "818535",
        "820896",
        "823635",
        "827381"
    ],
    "type_14": [
        "818596",
        "820955",
        "824442",
        "827526"
    ]
    }
  
def select_datasets_by_type(base_path, data_type='type_1'):

    datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    

    datasets.sort()
    selected_datasets = type_datasets_dict[data_type]    
    # selected_datasets = ['10002', '10016', '10022', '10023', '10027', '10031', '10036', '10057', '10058']  
    # selected_datasets = datasets[30:30+num_selected]

    print('selected_datasets', selected_datasets)
    
    return selected_datasets

def select_datasets_eval_top1(base_path, num_selected=1):

    datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    

    datasets.sort()

    selected_datasets = datasets[10:10+num_selected]        

    print('selected eval datasets', selected_datasets)
    
    return selected_datasets

def select_datasets(base_path, num_selected=10):

    datasets = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    

    datasets.sort()
    
    interval = max(1, len(datasets) // num_selected)
    
    selected_datasets = []
    for i in range(0, len(datasets), interval):
        if len(selected_datasets) < num_selected:
            selected_datasets.append(datasets[i])

    selected_datasets = selected_datasets[:num_selected]        

    print('selected_datasets', selected_datasets)
    
    return selected_datasets

if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_TRT_ENABLE"] = "false"
    os.environ["WANDB_DISABLE_CODE"] = "TRUE"
    os.environ["WANDB_MODE"] = "dryrun"
    torch.cuda.empty_cache()

    args = parse_args()
    if args.gym_id == "ecs-v2":
        args.data_list = select_datasets_by_type(args.base_path, data_type=args.type)
        args.eval_data_list = select_datasets_by_type(args.base_path, data_type=args.type)
    
    plot_every_step = 20
    test_every_step = 20
    best_eval_score = float('inf')
    num_envs = args.num_envs
    # num_steps = args.num_steps
    if args.gym_id == "ecs-v1":
        run_name = f"{args.gym_id}_{args.data_path}_{args.model}_{args.seed}" \
               f"_{utils.name_with_datetime()}"
    elif args.gym_id == "ecs-v2":
        run_name = f"new_type_{args.gym_id}_{args.type}_{args.model}_{args.seed}" \
           f"_{utils.name_with_datetime().replace(' ', '_').replace(':', '_').replace('-', '_')}"
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)
    if args.track:
        wandb.init(entity="zhykoties",
                   project="vm_scheduling",
                   name=run_name,
                   sync_tensorboard=True,
                   monitor_gym=True, config={
                'model': args.model,
                'ent_coef': args.ent_coef,
                'vf_coef': args.vf_coef,
                'eff_b_sz': args.accum_iter * args.num_minibatches},
                   # notes="",
                   tags=[args.model, args.gym_id, 'norm'] if args.normalize else [args.model, args.gym_id],
                   save_code=True
                   )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    print('vf_coef: ', args.vf_coef)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # env setup
    if args.gym_id == 'ecs-v1':
        envs = AsyncVectorEnv_Patch(
            [make_env(args.gym_id, args.seed, args.data_path) for i in range(num_envs)]
        )
        eval_envs = AsyncVectorEnv_Patch(
            [make_env(args.gym_id, args.seed, args.data_path) for i in range(num_envs)]
        )
    elif args.gym_id == 'ecs-v2':
        print("Training Envs Setting")
        envs = AsyncVectorEnv_Patch(
            [make_env(args.gym_id, args.seed, args.data_list, mode='train') for i in range(num_envs)]
        )
        print("Eval Envs Setting")
        eval_envs = AsyncVectorEnv_Patch(
            [make_env(args.gym_id, args.seed, args.eval_data_list, mode='eval') for i in range(num_envs)]
        )

    # args.num_steps = envs.env_fns[0]().configs['maxMigration']
    args.num_steps = envs.env_fns[0]().configs['maxMigration'] * 2
    # print("args.num_steps", args.num_steps)
    # args.num_steps = 100
    num_steps = args.num_steps
    num_test_steps = eval_envs.env_fns[0]().configs['maxMigration']  * 2
    args.batch_size = int(args.num_envs * args.num_steps) // 2
    args.minibatch_size = int(args.batch_size // (args.num_minibatches * args.accum_iter))

    

    params = utils.Params(f'./experiments/pretrain/{args.model}/params.json')
    params.update('./data/params.json')
    params.device = device
    params.batch_size = args.num_envs
    params.accum_iter = args.accum_iter

    params.num_vm = envs.num_items
    params.num_pm = envs.num_boxes

    print('clip_vloss: ', args.clip_vloss)

    # input the vm candidate model
    if args.model == 'attn':
        vm_cand_model = models.VM_Attn_Wrapper(params, args.pretrain).model
        pm_cand_model = models.PM_Attn_Wrapper(params).model
    elif args.model == 'mlp':
        vm_cand_model = models.VM_MLP_Wrapper(params, args.pretrain).model
        pm_cand_model = models.PM_MLP_Wrapper(params).model
    else:
        raise ValueError(f'args.model = {args.model} is not defined!')

    agent = Agent(envs, vm_cand_model, pm_cand_model, params, args.model)
    vm_optim = optim.Adam(vm_cand_model.parameters(), lr=args.learning_rate, eps=1e-5)
    pm_optim = optim.Adam(pm_cand_model.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.track:
        wandb.watch(agent, log_freq=100)

    # ALGO Logic: Storage setup
    obs_vm = torch.zeros(num_steps, args.num_envs, envs.num_items, params.vm_cov, device=device)
    obs_pm = torch.zeros(num_steps, args.num_envs, envs.num_boxes, params.pm_cov, device=device)
    obs_num_steps = torch.zeros(num_steps, args.num_envs, 1, 1, device=device)
    obs_num_vms = torch.zeros(num_steps, args.num_envs, dtype=torch.int32, device=device)
    vm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    vm_logprobs = torch.zeros(num_steps, args.num_envs, device=device)
    pm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    pm_logprobs = torch.zeros(num_steps, args.num_envs, device=device)
    rewards = torch.zeros(num_steps, args.num_envs, device=device)
    dones = torch.zeros(num_steps, args.num_envs, device=device)
    values = torch.zeros(num_steps, args.num_envs, device=device)
    # envs.single_action_space.nvec: [2089, 279] (#vm, #pm)
    action_masks = torch.zeros(num_steps, args.num_envs, envs.single_action_space.nvec[1], dtype=torch.bool,
                               device=device)
    vm_masks = torch.zeros(num_steps, args.num_envs, envs.num_items, dtype=torch.bool, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0

    if args.debug:
        col_names = ['step']
        for i in range(envs.num_items):
            for j in range(params.vm_cov):
                col_names.append(f'vm_{i}_cov_{j}')

        for i in range(envs.num_boxes):
            for j in range(params.pm_cov):
                col_names.append(f'pm_{i}_cov_{j}')

        col_names += ['num_steps', 'num_vms', 'vm_action', 'vm_logprob', 'pm_action', 'pm_logprob', 'rewards', 'done']
        col_names += ['values', 'ep_return', 'ep_cost']
        plot_step = np.tile(np.expand_dims(np.arange(num_steps), -1), 3).reshape((num_steps, 3, 1))

    num_updates = args.total_timesteps // args.batch_size
    save_every_step = num_updates // 8

    pbar = trange(1, num_updates + 1)
    for update in pbar:
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = args.learning_rate
            vm_optim.param_groups[0]["lr"] = lrnow
            pm_optim.param_groups[0]["lr"] = lrnow

        current_ep_info = np.zeros((num_steps, args.num_envs, 2)) - 1000  # return, len, fr
        current_ep_score = [] 

        next_obs_dict = envs.reset()
        next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
        next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
        next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
        next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
        next_done = torch.zeros(args.num_envs, device=device)


        for step in range(0, num_steps):
            global_step += 1 * args.num_envs

            obs_pm[step] = next_obs_pm
            obs_vm[step] = next_obs_vm
            obs_num_steps[step] = next_obs_num_steps
            obs_num_vms[step] = next_obs_num_vms
            dones[step] = next_done

            with torch.no_grad():
                vm_action, vm_logprob, _, pm_action, pm_logprob, _, value, action_mask, vm_mask \
                    = agent.get_action_and_value(envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)
                values[step] = value.flatten()  # value:  torch.Size([8, 1])
            action_masks[step] = action_mask
            vm_masks[step] = vm_mask
            vm_actions[step] = vm_action
            vm_logprobs[step] = vm_logprob
            pm_actions[step] = pm_action
            pm_logprobs[step] = pm_logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # print(f'vm_action: {vm_action.cpu().numpy()}, pm_action: {pm_action.cpu().numpy()}')
            next_obs_dict, reward, done, info = envs.step(torch.stack([vm_action, pm_action],
                                                                      dim=-1).cpu().numpy())
                                                             
            next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
            next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_done = torch.Tensor(done).to(device)

            for env_id, item in enumerate(info):
                if "episode" in item.keys():
                    current_ep_info[step, env_id, 0] = item["episode"]["r"]
                if done[env_id]:
                    current_ep_score.append(item["done_score"])
                    print(f"[Train]: env-id {env_id} is done! run {item['move_count']} steps, done score:{item['done_score']}")
        no_end_mask = current_ep_info[:, :, 0] != -1000
        avg_current_ep_score = np.mean(np.array(current_ep_score))
        current_ep_return = current_ep_info[:, :, 0][no_end_mask]
        if args.track:
            writer.add_scalar("episode_details/episodic_return", np.mean(current_ep_return), global_step)
            # writer.add_scalar("episode_details/episodic_return", avg_current_ep_return, global_step)
            writer.add_scalar("episode_details/episodic_score", avg_current_ep_score, global_step)

        pbar.set_description(f'Train avg episodic return: {np.mean(current_ep_return):.4f} | episodic score: {avg_current_ep_score:.4f}')
        # pbar.set_description(f'Train avg episodic return: {avg_current_ep_return:.4f} | episodic score: {avg_current_ep_score:.4f}')
        # if args.track:
        #     table = wandb.Table(data=np.stack([current_ep_return, current_ep_fr], axis=-1),
        #                         columns=["return", "fragment rate"])
        #     wandb.log({"episode_details/return_vs_FR": wandb.plot.scatter(table, "return", "fragment rate")})
        if args.debug:
            print(f'========= global_step: {global_step} ========= '
                  f'\n{np.stack([current_ep_return, current_ep_fr], axis=-1)}')

        if args.debug and (update + 1) % plot_every_step == 0:
            plot_obs_vm = obs_vm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
            plot_obs_pm = obs_pm[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
            plot_obs_num_steps = obs_num_steps[:, :3].cpu().data.numpy().reshape(num_steps, 3, -1)
            plot_obs_num_vms = obs_num_vms[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_vm_actions = vm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_vm_logprobs = vm_logprobs[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_pm_actions = pm_actions[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_pm_logprobs = pm_logprobs[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_rewards = rewards[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_dones = dones[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_values = values[:, :3].cpu().data.numpy().reshape(num_steps, 3, 1)
            plot_ep_info = current_ep_info[:, :3]
            plot_update_all = np.swapaxes(np.concatenate([plot_step, plot_obs_vm, plot_obs_pm, plot_obs_num_steps,
                                                          plot_obs_num_vms, plot_vm_actions, plot_vm_logprobs,
                                                          plot_pm_actions, plot_pm_logprobs, plot_rewards, plot_dones,
                                                          plot_values, plot_ep_info], axis=-1), axis1=1, axis2=0)
            plot_update_all = plot_update_all.reshape((num_steps * 3, -1))
            episode_df = pd.DataFrame(plot_update_all, columns=col_names)
            plot_fr_mean = np.mean(plot_ep_info[:, :, 2][plot_ep_info[:, :, 2] != -1000])
            episode_df.to_pickle(f'runs/{run_name}/u_{update}_{plot_fr_mean}.pkl')

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards, device=device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards, device=device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs_vm = obs_vm.reshape(-1, envs.num_items, params.vm_cov)
        b_obs_pm = obs_pm.reshape(-1, envs.num_boxes, params.pm_cov)
        b_obs_num_steps = obs_num_steps.reshape(-1, 1, 1)
        b_obs_num_vms = obs_num_vms.reshape(-1)
        b_vm_actions = vm_actions.reshape(-1)
        b_vm_logprobs = vm_logprobs.reshape(-1)
        b_pm_actions = pm_actions.reshape(-1)
        b_pm_logprobs = pm_logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_action_masks = action_masks.reshape(-1, envs.single_action_space.nvec[1])
        b_vm_masks = vm_masks.reshape(-1, envs.single_action_space.nvec[0])
        if args.debug:
            print('CRITIC CHECK - returns (pred vs real):\n',
                  torch.stack([b_values, b_returns], dim=-1).cpu().data.numpy()[:50])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        vm_clipfracs = []
        pm_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for index, start in enumerate(range(0, args.batch_size, args.minibatch_size)):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, vm_newlogprob, vm_entropy, _, pm_newlogprob, pm_entropy, newvalue, _, _ = agent.get_action_and_value(
                    envs,
                    b_obs_pm[mb_inds],
                    b_obs_vm[mb_inds],
                    b_obs_num_steps[mb_inds],
                    b_obs_num_vms[mb_inds],
                    vm_mask=b_vm_masks[mb_inds],
                    pm_mask=b_action_masks[mb_inds],
                    selected_vm=b_vm_actions.long()[mb_inds],
                    selected_pm=b_pm_actions.long()[mb_inds]
                )

                vm_logratio = vm_newlogprob - b_vm_logprobs[mb_inds]
                vm_ratio = vm_logratio.exp()
                pm_logratio = pm_newlogprob - b_pm_logprobs[mb_inds]
                pm_ratio = pm_logratio.exp()
                # if epoch == 0 and start == 0:
                #     print(f'pm_ratio: {pm_ratio}, vm_ratio: {vm_ratio}')

                with torch.no_grad():
                    vm_old_approx_kl = (-vm_logratio).mean()
                    vm_approx_kl = ((vm_ratio - 1) - vm_logratio).mean()
                    vm_clipfracs += [((vm_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    pm_old_approx_kl = (-pm_logratio).mean()
                    pm_approx_kl = ((pm_ratio - 1) - pm_logratio).mean()
                    pm_clipfracs += [((pm_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                vm_pg_loss1 = -mb_advantages.detach() * vm_ratio
                vm_pg_loss2 = -mb_advantages.detach() * torch.clamp(vm_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                vm_pg_loss = torch.max(vm_pg_loss1, vm_pg_loss2).mean()
                pm_pg_loss1 = -mb_advantages.detach() * pm_ratio
                pm_pg_loss2 = -mb_advantages.detach() * torch.clamp(pm_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pm_pg_loss = torch.max(pm_pg_loss1, pm_pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                vm_entropy_loss = vm_entropy.mean()
                pm_entropy_loss = pm_entropy.mean()
                vm_loss = (vm_pg_loss - args.ent_coef * vm_entropy_loss + v_loss * args.vf_coef) / params.accum_iter
                pm_loss = (pm_pg_loss - args.ent_coef * pm_entropy_loss) / params.accum_iter

                vm_loss.backward()
                pm_loss.backward()
                if ((index + 1) % params.accum_iter == 0) or (start + args.minibatch_size > args.batch_size):
                    nn.utils.clip_grad_norm_(agent.vm_net.parameters(), args.max_grad_norm)
                    vm_optim.step()
                    nn.utils.clip_grad_norm_(agent.pm_net.parameters(), args.max_grad_norm)
                    pm_optim.step()
                    vm_optim.zero_grad(set_to_none=True)
                    pm_optim.zero_grad(set_to_none=True)

            if args.target_kl is not None:
                if max(vm_approx_kl, pm_approx_kl) > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if update % test_every_step == 0:
            
            agent.eval()
            eval_ep_rewards = []
            eval_ep_score = []
            with torch.no_grad():
                for _ in range(eval_envs.num_eval_data):

                    next_obs_dict = eval_envs.reset()
                    next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                    next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                    next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                    next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)

                    for step in range(0, num_test_steps):
                        vm_action, vm_logprob, _, pm_action, pm_logprob, _, value, action_mask, vm_mask \
                            = agent.get_action_and_value(eval_envs, next_obs_pm, next_obs_vm, next_obs_num_steps, next_obs_num_vms)

                        next_obs_dict, reward, done, info = eval_envs.step(torch.stack([vm_action, pm_action],
                                                                                    dim=-1).cpu().numpy())
                        next_obs_pm = torch.Tensor(next_obs_dict['pm_info']).to(device)
                        next_obs_vm = torch.Tensor(next_obs_dict['vm_info']).to(device)
                        next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
                        next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
                        next_done = torch.Tensor(done).to(device)

                        for env_id, item in enumerate(info):
                            if "episode" in item.keys():
                                eval_ep_rewards.append(item["episode"]["r"])
                            if done[env_id]:
                                eval_ep_score.append(item["done_score"])
                                print(f"[Eval]: env-id {env_id} is done! run {item['move_count']} steps, done score:{item['done_score']}")
            if args.track:
                avg_eval_ep_score = np.mean(np.array(eval_ep_score))
                avg_eval_ep_rewards = np.mean(np.array(eval_ep_rewards))
                current_ep_return = current_ep_info[:, :, 0][no_end_mask]
                writer.add_scalar("Eval/eval_episodic_return", avg_eval_ep_rewards, global_step)
                writer.add_scalar("Eval/eval_episodic_score", avg_eval_ep_score, global_step)
                print(f"[Eval]:global_step:{global_step} | eval_episodic_return:{avg_eval_ep_rewards} | eval_episodic_score:{avg_eval_ep_score}")
                if avg_eval_ep_score < best_eval_score:
                    best_eval_score = avg_eval_ep_score
                    utils.save_checkpoint({
                        'global_step': global_step,
                        'state_dict': agent.state_dict(),
                        'vm_optim_dict': vm_optim.state_dict(),
                        'pm_optim_dict': pm_optim.state_dict(),
                        'best_score': best_eval_score
                    }, global_step=global_step, checkpoint=f"runs/{run_name}", is_best=True)
                    print(f"New best model saved with score: {best_eval_score}")
            agent.train()

        if args.track:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("Charts/vm_learning_rate", vm_optim.param_groups[0]["lr"], global_step)
            writer.add_scalar("Charts/pm_learning_rate", pm_optim.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/vm_policy_loss", vm_pg_loss.item(), global_step)
            writer.add_scalar("losses/vm_entropy", vm_entropy_loss.item(), global_step)
            writer.add_scalar("losses/vm_old_approx_kl", vm_old_approx_kl.item(), global_step)
            writer.add_scalar("losses/vm_approx_kl", vm_approx_kl.item(), global_step)
            writer.add_scalar("losses/vm_clipfrac", np.mean(vm_clipfracs), global_step)
            writer.add_scalar("losses/pm_policy_loss", pm_pg_loss.item(), global_step)
            writer.add_scalar("losses/pm_entropy", pm_entropy_loss.item(), global_step)
            writer.add_scalar("losses/pm_old_approx_kl", pm_old_approx_kl.item(), global_step)
            writer.add_scalar("losses/pm_approx_kl", pm_approx_kl.item(), global_step)
            writer.add_scalar("losses/pm_clipfrac", np.mean(pm_clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            if update % save_every_step == 0 or update == num_updates or global_step == args.total_timesteps * 2:
                utils.save_checkpoint({'global_step': global_step,
                                       'state_dict': agent.state_dict(),
                                       'vm_optim_dict': vm_optim.state_dict(),
                                       'pm_optim_dict': pm_optim.state_dict()},
                                      global_step=global_step,
                                      checkpoint=f"runs/{run_name}")
    utils.save_checkpoint({'global_step': global_step,
                                       'state_dict': agent.state_dict(),
                                       'vm_optim_dict': vm_optim.state_dict(),
                                       'pm_optim_dict': pm_optim.state_dict()},
                                      global_step=global_step,
                                      checkpoint=f"runs/{run_name}")
                                      
    envs.close()
    if args.track:
        writer.close()
