import os
import json
from datetime import datetime

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR

from mtl.datasets.dataset_miniscapes import DatasetMiniscapes
from mtl.models.model_deeplab_v3_plus import ModelDeepLabV3Plus
from mtl.models.model_branched_arch import ModelBranchedArch
from mtl.models.model_task_distillation import ModelTaskDistill
from mtl.models.model_task_distill_all_connect import ModelTaskDistillAllConnect
from mtl.models.model_task_distill_all_attend import ModelTaskDistillAllConnectAllAttend


def resolve_dataset_class(name):
    return {
        'miniscapes': DatasetMiniscapes,
    }[name]


def resolve_model_class(name):
    return {
        'deeplabv3p': ModelDeepLabV3Plus,
        'brancharch': ModelBranchedArch,
        'taskdistill': ModelTaskDistill, 
        'taskdistillallcon': ModelTaskDistillAllConnect,
        'taskdistillallconallattend': ModelTaskDistillAllConnectAllAttend
    }[name]


def resolve_optimizer(cfg, params):
    if cfg.optimizer == 'sgd':
        return SGD(
            params,
            lr=cfg.optimizer_lr,
            momentum=cfg.optimizer_momentum,
            weight_decay=cfg.optimizer_weight_decay,
        )
    elif cfg.optimizer == 'adam':
        return Adam(
            params,
            lr=cfg.optimizer_lr,
            weight_decay=cfg.optimizer_weight_decay,
        )
    else:
        raise NotImplementedError


def resolve_lr_scheduler(cfg, optimizer):
    if cfg.lr_scheduler == 'poly':
        return LambdaLR(
            optimizer,
            lambda ep: max(1e-6, (1 - ep / (cfg.num_epochs-1)) ** cfg.lr_scheduler_power)
        )
    else:
        raise NotImplementedError


def create_experiment_name(cfg):
    dtime = datetime.now().strftime("%y-%m-%d:%H-%M-%S")
    exp_name = f'{dtime}_b{cfg.batch_size}_lr{cfg.optimizer_lr}_{cfg.message}'

    # return exp_name
    cfg.log_dir = os.path.join(cfg.log_dir, exp_name)
    return cfg

def save_config(cfg):
    with open(os.path.join(cfg.log_dir, 'config.json'), 'w') as json_file:
        cfg_dict = json.dumps(cfg.__dict__, indent=4, sort_keys=True)
        json.dump(cfg_dict, json_file)