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
from tqdm import trange

import gym_reschdule_combination.envs.vm_rescheduler_env

import models
import utils
from env_patch import AsyncVectorEnv_Patch
from main_ecs import make_env

import re
import pandas as pd

# type_datasets_dict = {
#     "type_1": ['817062', '821576', '824526', '828295', '831438', '838812', '839913', '840007', '840008', '841180', '842516', '843693', '843702', '845597', '847814', '847829', '849590', '858874'],
#     "type_2": ['10003', '10027', '10056', '10065', '10140', '10187', '10506', '10752', '10839', '10875', '10908', '10931', '11349', '11368', '11509', '12027', '12067', '13675', '13691', '13710', '14061', '14133', '14172', '14206', '14236', '14328', '14370', '14494', '14614', '8621', '8975', '8994', '9075'],
#     "type_3": ['10002', '10016', '10022', '10023', '10031', '10036', '10057', '10058', '10062', '10086', '10095', '10114', '10124', '10156', '10158', '10167', '10177', '10221', '10514', '10726', '10778', '10813', '10848', '10881', '10898', '10954', '11344', '11367', '11529', '11834', '12055', '12541', '12571', '13006', '13324', '13651', '13658', '13659', '13665', '13680', '14045', '14047', '820624', '8616', '8660', '8700', '8767', '8858', '8971', '9002', '9094', '9938'],
#     "type_4": ['817807', '818598', '818659', '819461', '820119', '820957', '823608', '824384', '826435', '827469', '827590', '829940', '832566', '835928', '836820', '840118', '841287', '843634', '845695', '845761', '849721', '853124', '857687', '858881'],
#     "type_5": ['817026', '833587', '837830', '841263', '847716', '849591'],
#     "type_6": ['817868', '817871', '818604', '819305', '820839', '820963', '820964', '821518', '821629', '821650', '823562', '824360', '824438', '824527', '825451', '825526', '826267', '826422', '827478', '827588', '828375', '828377', '829110', '829111', '831326', '834862', '836735', '841271', '841279', '854467', '854476', '858863'],
#     "type_7": ['817026', '818603', '819988', '820968', '821647', '825501', '828212', '832472', '833587', '837830', '840006', '841263', '842600', '843701', '847716', '847735', '849591', '858875'],
#     "type_8": ['10784', '10809', '11778', '11833', '13010', '13650', '13727', '13741', '821464', '8674', '8706', '8780', '8828', '8832', '8855'],
#     # "type_9":['816943', '817006', '817069', '817076', '817731', '817806', '817808', '817855', '817868', '817871', '817879', '818604', '818618', '818644', '818649', '819254', '819257', '819305', '819316', '819371', '819379', '819974', '819982', '820041', '820099', '820103', '820839', '820842', '820962', '820963', '820964', '820991', '821518', '821575', '821580', '821629', '821633', '821650', '821680', '823437', '823533', '823535', '823562', '824360', '824438', '824517', '824520', '824527', '825329', '825415', '825420', '825437', '825451', '825503', '825510', '825515', '825526', '826267', '826269', '826408', '826410', '826422', '826429', '827478', '827588', '827662', '828209', '828293', '828299', '828306', '828355', '828356', '828357', '828358', '828375', '828377', '828391', '829019', '829110', '829111', '829954', '831326', '831327', '831419', '831444', '832385', '832469', '832556', '834862', '834875', '835796', '835848', '836603', '836735', '838811', '838902', '838903', '838985', '839013', '840011', '840095', '840106', '840110', '841271', '841279', '842511', '842591', '842592', '843503', '843677', '844729', '845746', '845750', '847674', '847737', '847754', '847755', '849621', '849807', '851684', '851690', '851692', '853064', '854467', '854476', '854485', '855533', '855594', '858796', '858850', '858863', '858872', '858885'],
#     # "type_10":['10784', '10809', '11778', '11833', '13010', '13317', '13650', '13727', '13741', '821464', '8674', '8706', '8780', '8828', '8832', '8855'],
#     }

# type_datasets_dict = {
#     "type_1": ['10003', '10027', '10056', '10065', '10104', '10112', '10135', '10140', '10187', '10506', '10752', '10784', '10809', '10839', '10875', '10908', '10931', '11349', '11368', '11509', '11778', '11833', '12027', '12067', '12553', '12574', '13010', '13317', '13650', '13675', '13691', '13710', '13727', '13741', '14061', '14133', '14172', '14206', '14236', '14328', '14370', '14409', '14494', '14614', '14633', '14642', '821464', '8621', '8674', '8706', '8780', '8828', '8832', '8855', '8975', '8994', '9075'],
#     "type_2": ['10002', '10016', '10022', '10023', '10031', '10036', '10057', '10058', '10062', '10086', '10095', '10114', '10124', '10156', '10158', '10167', '10177', '10221', '10514', '10726', '10778', '10813', '10848', '10881', '10898', '10954', '11344', '11367', '11529', '11834', '12055', '12541', '12571', '13006', '13324', '13651', '13658', '13659', '13665', '13680', '14045', '820624', '8616', '8660', '8700', '8767', '8858', '8971', '9002', '9094', '9938'],
#     "type_3": ['821576', '824526', '828295', '831438', '840008', '843693', '847814', '849322', '849342', '858279', '858285', '858303', '858408', '858493', '858534', '858588'],
#     "type_4": ['841935', '842308', '842479', '842697', '843112', '843213', '844254', '844308', '844320', '844357', '845479', '845528', '845633', '845983', '846083', '846175', '846340', '846412', '846513', '846525', '846621', '846803', '847043', '847216', '847275', '847393', '847453', '847462', '847476', '856897'],
#     "type_5": ['819257', '821575', '824520', '828209', '831327', '834875', '838811', '842511', '845746', '849621', '853064'],
#     "type_6": ['817879', '820978', '824541', '828390', '832572', '836830', '841291', '845764', '849830', '854485', '858885'],
#     "type_7": ['817078', '820112', '823603', '839907', '843702', '852760', '852815', '852889', '852927', '853141'],
#     "type_8": ['823534', '824441', '825425', '826348', '827383', '828298', '829108', '829861', '831332', '832479', '833517', '834786', '836609', '837657', '838812', '839913', '841180', '842516', '843509', '844531', '845602', '846758', '847708', '857621', '858740'],
#     "type_9": ['817807', '820102', '823530', '826414', '829863', '833586', '837759', '841265', '845695', '849721', '853124', '857687'],
#     "type_10": ['818433', '818453', '818469', '818472', '818474', '818476', '818479', '818483', '818582', '818914', '824513', '828309'],
#     }
    

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

directory = "/mnt/workspace/workgroup/xuwan/datas/novio_finite_noii_v1"

subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
print("all subdirs", subdirs)
type_datasets_dict = {
    "type_all": subdirs,
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

def parse_args():
    # fmt: offpython eval_ecs.py --restore-name ecs-v2_type_3_mlp_1_2024_09_02_09_27_01 --restore-file-name 1999500 --type 'type_3'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp", help="model architecture")
    parser.add_argument("--restore-name", type=str, required=True, help="restore experiment name")
    parser.add_argument("--restore-file-name", type=str, required=True, help="restore file name")
    parser.add_argument("--pretrain", action='store_true',
                        help="if toggled, we will restore pretrained weights for vm selection")
    parser.add_argument("--gym-id", type=str, default="ecs-v2",
                        help="the id of the gym environment")
    parser.add_argument("--type", type=str, default="type_2",
                        help="the type of test datasets")
    parser.add_argument("--base-path", type=str, default="/mnt/workspace/DRL-based-VM-Rescheduling/data/nocons_finite_easy", help="ecs data path")
    parser.add_argument("--data-list", type=list, default=['8616'], help="ecs data path")

    parser.add_argument("--vm-data-size", type=str, default="M", choices=["M", "L"],
                        help="size of the dataset")
    parser.add_argument("--max-steps", type=int, default=50, help="maximum number of redeploy steps")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
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
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--accum-iter", type=int, default=4,
                        help="number of iterations where gradient is accumulated before the weights are updated;"
                             " used to increase the effective batch size")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.005,  # 0.01
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1e-2,  # 1e-4
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // (args.num_minibatches * args.accum_iter))
    return args


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

        if vm_mask is None:
            vm_mask = torch.tensor(np.array(envs.call_parse('get_vm_dense_mask', id=[1] * args.num_envs)), dtype=torch.bool, device=self.device)  # pm_mask:  torch.Size([8, 279])

        b_sz = obs_info_pm.shape[0]



        if self.model == "attn":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_num_steps, obs_info_pm, vm_mask)
        elif self.model == "mlp":
            vm_logits, critic_score = self.vm_net(obs_info_all_vm, obs_info_pm)
        else:
            raise ValueError(f'self.model={self.model} is not implemented')

        vm_cat = CategoricalMasked(logits=vm_logits, masks=vm_mask)
        if selected_vm is None:
            # if vm_cat.probs.max() <= 0.1:
            #     selected_vm = vm_cat.sample()
            # # selected_vm = vm_cat.sample()
            # # print("vm_cat.probs", vm_cat.probs.max())
            # else:
            selected_vm = torch.argmax(vm_cat.probs, dim=1).long()
        vm_log_prob = vm_cat.log_prob(selected_vm)

        if pm_mask is None:
            pm_mask = torch.tensor(np.array(envs.call_parse('get_pm_mask', item_id=selected_vm.cpu().tolist())),
                                   dtype=torch.bool, device=self.device)  # pm_mask:  torch.Size([8, 279])

        if "attn" in self.model:
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_num_steps,
                                    obs_info_pm)  # b_sz
        elif self.model == "mlp":
            pm_logits = self.pm_net(obs_info_all_vm[torch.arange(b_sz), selected_vm].unsqueeze(1), obs_info_pm)  # b_sz
        else:
            raise ValueError(f'self.model={self.model} is not implemented')

        pm_cat = CategoricalMasked(logits=pm_logits, masks=pm_mask)
        if selected_pm is None:
            # selected_pm = pm_cat.sample()  # selected_pm:  torch.Size([8])
            # print("pm_cat.probs", pm_cat.probs.max())
            selected_pm = torch.argmax(pm_cat.probs, dim=1).long()
        pm_log_prob = pm_cat.log_prob(selected_pm)  # pm_log_prob:  torch.Size([8])

        return selected_vm, vm_log_prob, vm_cat.entropy(), selected_pm, pm_log_prob, \
               pm_cat.entropy(), critic_score, pm_mask, vm_mask

def select_datasets_by_type(base_path, data_type='tyep_1'):

    selected_datasets = type_datasets_dict[data_type]    

    print('selected_datasets', selected_datasets)
    
    return selected_datasets

if __name__ == "__main__":
    args = parse_args()
    if args.gym_id == "ecs-v2":
        args.data_list = select_datasets_by_type(args.base_path, data_type=args.type)

    results = {
    'dataset_id': [], 
    'rl_score': [], 
    'rl_rewards': [], 
    'greedy_score': [], 
    'greedy_rewards': [], 
    'base_score': [], 
    'base_rewards': [], 
    'evaluation_time': []
    }
    args.num_envs = 1
    args.seed = 1
    num_envs = args.num_envs
    run_name = f'{args.restore_name}'
    np.set_printoptions(precision=4)
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)
    print('vf_coef: ', args.vf_coef)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = AsyncVectorEnv_Patch(
            [make_env(args.gym_id, args.seed, args.data_list, mode='test') for i in range(num_envs)]
        )


    args.num_steps = envs.env_fns[0]().configs['maxMigration']
    num_steps = args.num_steps
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
    # optim = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    agent.eval()
    global_step, agent = utils.load_checkpoint(args.restore_name, args.restore_file_name, agent)
    print(f"- Restored file (global step {global_step}) "
          f"from {os.path.join(args.restore_name, args.restore_file_name + '.pth.tar')}")

    if args.track:
        wandb.watch(agent, log_freq=100)

    # ALGO Logic: Storage setup
    obs_vm = torch.zeros(num_steps, args.num_envs, params.num_vm, params.vm_cov, device=device)
    obs_pm = torch.zeros(num_steps, args.num_envs, params.num_pm, params.pm_cov, device=device)
    obs_num_steps = torch.zeros(num_steps, args.num_envs, 1, 1, device=device)
    obs_num_vms = torch.zeros(num_steps, args.num_envs, dtype=torch.int32, device=device)
    vm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    pm_actions = torch.zeros(num_steps, args.num_envs, device=device)
    logprobs = torch.zeros(num_steps, args.num_envs, device=device)
    rewards = torch.zeros(num_steps, args.num_envs, device=device)
    dones = torch.zeros(num_steps, args.num_envs, device=device)
    values = torch.zeros(num_steps, args.num_envs, device=device)

    action_masks = torch.zeros(num_steps, args.num_envs, envs.single_action_space.nvec[1], dtype=torch.bool, device=device)
    vm_masks = torch.zeros(num_steps, args.num_envs, envs.single_action_space.nvec[0], dtype=torch.bool, device=device)


    with torch.no_grad():
        test_dataset_type = args.type
        for file_index in range(len(type_datasets_dict[test_dataset_type])):
            start_time = time.time() 
            envs.call_parse('set_current_env', env_id=[file_index] * args.num_envs)
            results['dataset_id'].append(type_datasets_dict[test_dataset_type][file_index]) 

            current_ep_score = []
            current_ep_rewards = []

            greedy_log_path = f'/mnt/workspace/DRL-based-VM-Rescheduling/baselines/greedy_nocons_finite_easy/{type_datasets_dict[test_dataset_type][file_index]}.log'
            greedy_reward = None
            greedy_score = None

            base_log_path = f'/mnt/workspace/DRL-based-VM-Rescheduling/baselines/nocons_finite_easy/{type_datasets_dict[test_dataset_type][file_index]}.log'
            base_reward = None
            base_score = None

            reward_pattern = r'Migration Cost:\s*([-+]?\d*\.\d+|\d+)'
            score_pattern = r'Finish Score:\s*([-+]?\d*\.\d+|\d+)'

            base_reward_pattern = r'Total Reward:\s*([-+]?\d*\.\d+|\d+)'
            base_score_pattern = r'Final Score: \s*([-+]?\d*\.\d+|\d+)'

            if os.path.exists(greedy_log_path):
                with open(greedy_log_path, 'r') as log_file:
                    for line in log_file:
                        reward_match = re.search(reward_pattern, line)
                        if reward_match:
                            greedy_reward = float(reward_match.group(1))
                        
                        score_match = re.search(score_pattern, line)
                        if score_match:
                            greedy_score = float(score_match.group(1))

            results['greedy_rewards'].append(greedy_reward if greedy_reward is not None else 0)  
            results['greedy_score'].append(greedy_score if greedy_score is not None else 0) 

            if os.path.exists(base_log_path):
                with open(base_log_path, 'r') as log_file:
                    for line in log_file:
                        reward_match = re.search(base_reward_pattern, line)
                        if reward_match:
                            base_reward = float(reward_match.group(1))
                        
                        score_match = re.search(base_score_pattern, line)
                        if score_match:
                            base_score = float(score_match.group(1))

            results['base_rewards'].append(base_reward if base_reward is not None else 0)  
            results['base_score'].append(base_score if base_score is not None else 0) 

            current_ep_info = np.zeros((num_steps, args.num_envs, 2)) - 1000  # return, len, fr
            next_obs_dict = envs.reset()
            next_obs_pm = torch.tensor(next_obs_dict['pm_info'], device=device)  # torch.Size([8, 279, 8])
            next_obs_vm = torch.tensor(next_obs_dict['vm_info'], device=device)  # torch.Size([8, 279, 14])
            next_obs_num_steps = torch.Tensor(next_obs_dict['num_steps']).to(device)
            next_obs_num_vms = torch.tensor(next_obs_dict['num_vms'], dtype=torch.int32, device=device)
            next_done = torch.zeros(args.num_envs, device=device)

            for step in range(0, num_steps):

                # obs_pm[step] = next_obs_pm
                # obs_vm[step] = next_obs_vm
                # obs_num_steps[step] = next_obs_num_steps
                # obs_num_vms[step] = next_obs_num_vms
                # dones[step] = next_done

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
                pm_actions[step] = pm_action
                # logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
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
                        current_ep_rewards.append(item["done_rewards"])
                        
                        print(f"[Test]: env-id {env_id} is done! run {item['move_count']} steps, done score:{item['done_score']}")
                        break
                if np.any(done):
                    break
            avg_current_ep_score = np.mean(np.array(current_ep_score))
            avg_current_ep_rewards = np.mean(np.array(current_ep_rewards))
            
            no_end_mask = current_ep_info[:, :, 0] != -1000
            current_ep_return = current_ep_info[:, :, 0][no_end_mask]
            results['rl_rewards'].append(avg_current_ep_rewards)
            results['rl_score'].append(avg_current_ep_score)

            # print(f"[SCORE] greedy: {greedy_score} | rl: {np.mean(all_test_score)} | base: {base_score}")

            end_time = time.time() 
            evaluation_time = end_time - start_time 
            results['evaluation_time'].append(evaluation_time)

        df_results = pd.DataFrame(results)

        csv_filename = f"./test_all_metrics_{args.gym_id}_{args.type}_{args.model}_{args.seed}" \
           f"_{utils.name_with_datetime().replace(' ', '_').replace(':', '_').replace('-', '_')}.csv"
        df_results.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")

    envs.close()
