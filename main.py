##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You
## Modified by: RainbowSecret, JingyiXie, LangHuang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2020
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import time
import pdb

import torch
import torch.backends.cudnn as cudnn

from lib.utils.tools.logger import Logger as Log
from lib.utils.tools.configer import Configer
import optuna
import gc
import sqlite3
import numpy as np
def run(configer,trial):

    if configer.get('phase') == 'train' and configer.get("network","optuna"):
        configer.update(["train","batch_size"],trial.suggest_int("batch_size",2*torch.cuda.device_count(),configer.get("max_batch_size"),step=1))

        configer.update(["lr","base_lr"],trial.suggest_float("base_lr",0.0001, 0.0019, step=0.0003))
        configer.update(["lr","lr_policy"],trial.suggest_categorical("lr_policy",["step","lambda_poly"]))

        configer.get("lr","step")["gamma"] = trial.suggest_float("gamma",0.3,0.9,step=0.2)
        configer.get("lr","step")["step_size"] = trial.suggest_int("step_size",50,400,step=50)

        configer.update(["optim","optim_method"],trial.suggest_categorical("optim_method", ["sgd","adam"]))
        configer.update(["lr","nbb_mult"],trial.suggest_float("nbb_mult",0.25,4,step=0.25))

        if configer.get("optim","optim_method") == "sgd":
            configer.get("optim","sgd")["weight_decay"] = trial.suggest_float("weight_decay",0.00001, 0.0002, step=0.00003)
            configer.get("optim","sgd")["nesterov"] = trial.suggest_categorical("nesterov",[False,True])

        else:
            configer.get("optim","adam")["weight_decay"] = trial.suggest_float("weight_decay",0.0001, 0.0010, step=0.0002)

        configer.get("network","loss_weights")["aux_loss"] = trial.suggest_float("aux_loss",0.2, 0.8, step=0.2)
        configer.get("network","loss_weights")["seg_loss"] = trial.suggest_float("seg_loss",.5, 2.0, step=0.5)

        if configer.get("network","use_teach"):
            configer.update(["teacher_temp"],trial.suggest_float("teacher_temp", 1, 21, step=5))
            configer.update(["teacher_interp"],trial.suggest_float("teacher_interp", 0.1, 1, step=0.1))
        else:
            configer.update(["teacher_interp"],0)

        lossType = trial.suggest_categorical("loss_type", ["fs_auxce_loss","fs_auxohemce_loss "])
        lossType = lossType.replace(" ","")
        configer.update(["loss","loss_type"],lossType)

    if not configer.exists("trial_nb"):
        configer.add(["trial_nb"],trial.number)
    else:
        configer.update(["trial_nb"],trial.number)

    model = None
    if configer.get('method') == 'fcn_segmentor':
        if configer.get('phase') == 'train':
            from segmentor.trainer import Trainer
            model = Trainer(configer)
        elif configer.get('phase') == 'test':
            from segmentor.tester import Tester
            model = Tester(configer)
        elif configer.get('phase') == 'test_offset':
            from segmentor.tester_offset import Tester
            model = Tester(configer)
    else:
        Log.error('Method: {} is not valid.'.format(configer.get('task')))
        exit(1)

    if configer.get('phase') == 'train':
        miou = model.train(trial)
        return miou

    elif configer.get('phase').startswith('test') and configer.get('network', 'resume') is not None:
        model.test()
    else:
        Log.error('Phase: {} is not valid.'.format(configer.get('phase')))
        exit(1)


def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def getBestTrial(exp_id,model_id,trialNb=None):
    print("results/{}/{}_hypSearch.db".format(exp_id,model_id))
    con = sqlite3.connect("results/{}/{}_hypSearch.db".format(exp_id,model_id))
    curr = con.cursor()

    curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
    query_res = curr.fetchall()

    query_res = list(filter(lambda x:not x[1] is None,query_res))

    trialIds = [id_value[0] for id_value in query_res]
    values = [id_value[1] for id_value in query_res]

    if not trialNb is None:
        trialIds = trialIds[:trialNb]
        values = values[:trialNb]

    bestTrial = trialIds[np.array(values).argmax()]

    return bestTrial

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', default=None, type=str,
                        dest='configs', help='The file of the hyper parameters.')
    parser.add_argument('--phase', default='train', type=str,
                        dest='phase', help='The phase of module.')
    parser.add_argument('--gpu', default=[0,1,2,3], nargs='+', type=int,
                        dest='gpu', help='The gpu list used.')

    # ***********  Params for data.  **********
    parser.add_argument('--data_dir', default=None, type=str, nargs='+',
                        dest='data:data_dir', help='The Directory of the data.')
    parser.add_argument('--include_val', type=str2bool, nargs='?', default=False,
                        dest='data:include_val', help='Include validation set for training.')
    # include-coarse is only provided for Cityscapes.
    parser.add_argument('--include_coarse', type=str2bool, nargs='?', default=False,
                        dest='data:include_coarse', help='Include coarse-labeled set for training.')
    parser.add_argument('--only_coarse', type=str2bool, nargs='?', default=False,
                        dest='data:only_coarse', help='Only include coarse-labeled set for training.')
    parser.add_argument('--only_mapillary', type=str2bool, nargs='?', default=False,
                        dest='data:only_mapillary', help='Only include mapillary set for training.')
    parser.add_argument('--only_small', type=str2bool, nargs='?', default=False,
                        dest='data:only_small', help='Only include small val set for testing.')
    # include-atr is used to choose ATR as extra training set for LIP dataset.
    parser.add_argument('--include_atr', type=str2bool, nargs='?', default=False,
                        dest='data:include_atr', help='Include atr set for LIP training.')
    parser.add_argument('--include_cihp', type=str2bool, nargs='?', default=False,
                        dest='data:include_cihp', help='Include cihp set for LIP training.')
    parser.add_argument('--drop_last', type=str2bool, nargs='?', default=False,
                        dest='data:drop_last', help='Fix bug for syncbn.')
    parser.add_argument('--workers', default=None, type=int,
                        dest='data:workers', help='The number of workers to load data.')
    parser.add_argument('--train_batch_size', default=None, type=int,
                        dest='train:batch_size', help='The batch size of training.')
    parser.add_argument('--val_batch_size', default=None, type=int,
                        dest='val:batch_size', help='The batch size of validation.')

    # ***********  Params for checkpoint.  **********
    parser.add_argument('--checkpoints_root', default=None, type=str,
                        dest='checkpoints:checkpoints_root', help='The root dir of model save path.')
    parser.add_argument('--checkpoints_name', default=None, type=str,
                        dest='checkpoints:checkpoints_name', help='The name of checkpoint model.')
    parser.add_argument('--save_iters', default=None, type=int,
                        dest='checkpoints:save_iters', help='The saving iters of checkpoint model.')
    parser.add_argument('--save_epoch', default=None, type=int,
                        dest='checkpoints:save_epoch', help='The saving epoch of checkpoint model.')

    # ***********  Params for model.  **********
    parser.add_argument('--model_name', default=None, type=str,
                        dest='network:model_name', help='The name of model.')
    parser.add_argument('--backbone', default=None, type=str,
                        dest='network:backbone', help='The base network of model.')
    parser.add_argument('--bn_type', default=None, type=str,
                        dest='network:bn_type', help='The BN type of the network.')
    parser.add_argument('--multi_grid', default=None, nargs='+', type=int,
                        dest='network:multi_grid', help='The multi_grid for resnet backbone.')
    parser.add_argument('--pretrained', type=str, default=None,
                        dest='network:pretrained', help='The path to pretrained model.')
    parser.add_argument('--resume', default=None, type=str,
                        dest='network:resume', help='The path of checkpoints.')
    parser.add_argument('--resume_strict', type=str2bool, nargs='?', default=True,
                        dest='network:resume_strict', help='Fully match keys or not.')
    parser.add_argument('--resume_continue', type=str2bool, nargs='?', default=False,
                        dest='network:resume_continue', help='Whether to continue training.')
    parser.add_argument('--resume_eval_train', type=str2bool, nargs='?', default=True,
                        dest='network:resume_train', help='Whether to validate the training set  during resume.')
    parser.add_argument('--resume_eval_val', type=str2bool, nargs='?', default=True,
                        dest='network:resume_val', help='Whether to validate the val set during resume.')
    parser.add_argument('--gathered', type=str2bool, nargs='?', default=True,
                        dest='network:gathered', help='Whether to gather the output of model.')
    parser.add_argument('--loss_balance', type=str2bool, nargs='?', default=False,
                        dest='network:loss_balance', help='Whether to balance GPU usage.')

    parser.add_argument('--freeze_feature', type=str2bool, default=False,
                        dest='network:freeze_feature', help='To freeze the feature extraction model.')

    # ***********  Params for solver.  **********
    parser.add_argument('--optim_method', default=None, type=str,
                        dest='optim:optim_method', help='The optim method that used.')
    parser.add_argument('--group_method', default=None, type=str,
                        dest='optim:group_method', help='The group method that used.')
    parser.add_argument('--base_lr', default=None, type=float,
                        dest='lr:base_lr', help='The learning rate.')
    parser.add_argument('--nbb_mult', default=1.0, type=float,
                        dest='lr:nbb_mult', help='The not backbone mult ratio of learning rate.')
    parser.add_argument('--lr_policy', default=None, type=str,
                        dest='lr:lr_policy', help='The policy of lr during training.')
    parser.add_argument('--loss_type', default=None, type=str,
                        dest='loss:loss_type', help='The loss type of the network.')
    parser.add_argument('--is_warm', type=str2bool, nargs='?', default=False,
                        dest='lr:is_warm', help='Whether to warm training.')

    parser.add_argument('--use_teach', type=str2bool,default=False,
                        dest="network:use_teach")
    parser.add_argument('--interp', type=str2bool,default=False,
                        dest="network:interp")

    # ***********  Params for display.  **********
    parser.add_argument('--max_epoch', default=None, type=int,
                        dest='solver:max_epoch', help='The max epoch of training.')
    parser.add_argument('--max_iters', default=None, type=int,
                        dest='solver:max_iters', help='The max iters of training.')
    parser.add_argument('--display_iter', default=None, type=int,
                        dest='solver:display_iter', help='The display iteration of train logs.')
    parser.add_argument('--test_interval', default=None, type=int,
                        dest='solver:test_interval', help='The test interval of validation.')

    # ***********  Params for logging.  **********
    parser.add_argument('--logfile_level', default=None, type=str,
                        dest='logging:logfile_level', help='To set the log level to files.')
    parser.add_argument('--stdout_level', default=None, type=str,
                        dest='logging:stdout_level', help='To set the level to print to screen.')
    parser.add_argument('--log_file', default=None, type=str,
                        dest='logging:log_file', help='The path of log files.')
    parser.add_argument('--rewrite', type=str2bool, nargs='?', default=True,
                        dest='logging:rewrite', help='Whether to rewrite files.')
    parser.add_argument('--log_to_file', type=str2bool, nargs='?', default=True,
                        dest='logging:log_to_file', help='Whether to write logging into files.')

    # ***********  Params for test or submission.  **********
    parser.add_argument('--test_img', default=None, type=str,
                        dest='test:test_img', help='The test path of image.')
    parser.add_argument('--test_dir', default=None, type=str,
                        dest='test:test_dir', help='The test directory of images.')
    parser.add_argument('--out_dir', default='none', type=str,
                        dest='test:out_dir', help='The test out directory of images.')
    parser.add_argument('--save_prob', type=str2bool, nargs='?', default=False,
                        dest='test:save_prob', help='Save the logits map during testing.')

    # ***********  Params for env.  **********
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--cudnn', type=str2bool, nargs='?', default=True, help='Use CUDNN.')

    # ***********  Params for distributed training.  **********
    parser.add_argument('--local_rank', type=int, default=-1, dest='local_rank', help='local rank of current process')
    parser.add_argument('--distributed', action='store_true', dest='distributed', help='Use multi-processing training.')
    parser.add_argument('--use_ground_truth', action='store_true', dest='use_ground_truth', help='Use ground truth for training.')

    parser.add_argument('--exp_id', type=str,default="cityscapes")
    parser.add_argument('--max_batch_size', type=int,default=10)
    parser.add_argument('--optuna_trial_nb', type=int,default=300)

    parser.add_argument('--val_on_test', type=str2bool,default=False)
    parser.add_argument('--rep_vec', type=str2bool,default=True)

    parser.add_argument('--gradcam', type=str2bool,default=True,dest='test:gradcam')

    parser.add_argument('--pretr', type=str2bool,default=True,dest='network:pretr')
    parser.add_argument('--only_back_pretr', type=str2bool,default=False,dest='network:only_back_pretr')

    parser.add_argument('--optuna', type=str2bool,default=True,dest='network:optuna')

    parser.add_argument('--teacher_temp', type=float,default=10,dest='teacher_temp')
    parser.add_argument('--teacher_interp', type=float,default=0,dest='teacher_interp')

    parser.add_argument('REMAIN', nargs='*')

    args_parser = parser.parse_args()

    from lib.utils.distributed import handle_distributed
    handle_distributed(args_parser, os.path.expanduser(os.path.abspath(__file__)))

    if args_parser.seed is not None:
        random.seed(args_parser.seed)
        torch.manual_seed(args_parser.seed)

    cudnn.enabled = True
    cudnn.benchmark = args_parser.cudnn

    configer = Configer(args_parser=args_parser)
    data_dir = configer.get('data', 'data_dir')
    if isinstance(data_dir, str):
        data_dir = [data_dir]
    abs_data_dir = [os.path.expanduser(x) for x in data_dir]
    configer.update(['data', 'data_dir'], abs_data_dir)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    configer.add(['project_dir'], project_dir)

    if configer.get('logging', 'log_to_file'):
        log_file = configer.get('logging', 'log_file')
        new_log_file = '{}_{}'.format(log_file, time.strftime("%Y-%m-%d_%X", time.localtime()))
        configer.update(['logging', 'log_file'], new_log_file)
    else:
        configer.update(['logging', 'logfile_level'], None)

    Log.init(logfile_level=configer.get('logging', 'logfile_level'),
             stdout_level=configer.get('logging', 'stdout_level'),
             log_file=configer.get('logging', 'log_file'),
             log_format=configer.get('logging', 'log_format'),
             rewrite=configer.get('logging', 'rewrite'))

    def objective(trial):
        return run(configer,trial=trial)

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/{}".format(configer.get("exp_id"))):
        os.makedirs("results/{}".format(configer.get("exp_id")))

    study = optuna.create_study(direction="maximize",\
                                storage="sqlite:///results/{}/{}_hypSearch.db".format(configer.get("exp_id"),configer.get("checkpoints","checkpoints_name")), \
                                study_name=configer.get("checkpoints","checkpoints_name"),load_if_exists=True)

    con = sqlite3.connect("results/{}/{}_hypSearch.db".format(configer.get("exp_id"),configer.get("checkpoints","checkpoints_name")))
    curr = con.cursor()

    failedTrials = 0
    for elem in curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall():
        if elem[1] is None:
            failedTrials += 1

    trialsAlreadyDone = len(curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall())

    if trialsAlreadyDone-failedTrials < configer.get("optuna_trial_nb"):

        studyDone = False
        while not studyDone:
            try:
                print("N trials left",configer.get("optuna_trial_nb")-trialsAlreadyDone+failedTrials)
                study.optimize(objective,n_trials=configer.get("optuna_trial_nb")-trialsAlreadyDone+failedTrials)
                studyDone = True
            except RuntimeError as e:
                print("------- Max batch size was {} -------".format(configer.get("max_batch_size")))
                if str(e).find("CUDA out of memory.") != -1:
                    gc.collect()
                    torch.cuda.empty_cache()
                    old_max_bs = configer.get("max_batch_size")
                    configer.update(["max_batch_size"],old_max_bs-1)
                else:
                    raise RuntimeError(e)

    from segmentor.trainer import Trainer

    Log.info("self.configer.get('network','interp') {}".format(configer.get("network","interp")))

    model_id = configer.get("network","model_name")
    bestTrial = getBestTrial("cityscapes",model_id+"_")
    configer.add(["trial_nb"],bestTrial)
    bestWeights = "checkpoints/cityscapes/{}__trial{}_max_performance.pth".format(model_id,bestTrial-1)
    dic = torch.load(bestWeights)

    configer.add(["teacher_interp"],0)
    model = Trainer(configer)
    model.configer.resume(dic['config_dict'])
    model.seg_net.load_state_dict(dic["state_dict"])

    Log.info("loading {}".format(bestWeights))

    model.configer.add(["teacher_interp"],0)
    model.configer.add(["network","interp"],False)
    model.val(model.data_loader.get_valloader(dataset='val'),retSimMap=True,printallIoU=True,endEval=True,gradcam=configer.get("test","gradcam"))
