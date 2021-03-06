##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret, JingyiXie, LangHuang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import os
import cv2
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from lib.utils.tools.average_meter import AverageMeter
from lib.datasets.data_loader import DataLoader
from lib.loss.loss_manager import LossManager
from lib.models.model_manager import ModelManager
from lib.utils.tools.logger import Logger as Log
from lib.vis.seg_visualizer import SegVisualizer
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from segmentor.tools.data_helper import DataHelper
from segmentor.tools.evaluator import get_evaluator
from lib.utils.distributed import get_world_size, get_rank, is_distributed
import cv2
import sys

from gradcam import GradCAM,GradCAMpp
from guided_backprop import GuidedBackprop

class NoSplit():
    def __init__(self,obj):
        self.obj = obj

class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.foward_time = AverageMeter()
        self.backward_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_visualizer = SegVisualizer(configer)
        self.loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.model_manager = ModelManager(configer)
        self.data_loader = DataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)
        self.data_helper = DataHelper(configer, self)
        self.evaluator = get_evaluator(configer, self)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.running_score = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        if self.configer.get("network","freeze_feature"):
            self.seg_net.module.backbone.requires_grad = False

        if self.configer.get("teacher_interp") > 0:
            self.teach = self.model_manager.semantic_segmentor(teach=True)
            self.teach = self.module_runner.load_net(self.teach)
            self.teach.eval()
            self.teachDic = {}

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.train_loader = self.data_loader.get_trainloader()
        self.val_loader = self.data_loader.get_valloader("val")
        self.pixel_loss = self.loss_manager.get_seg_loss()
        if is_distributed():
            self.pixel_loss = self.module_runner.to_device(self.pixel_loss)

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self,trial):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        self.pixel_loss.train()
        start_time = time.time()

        if "swa" in self.configer.get('lr', 'lr_policy'):
            normal_max_iters = int(self.configer.get('solver', 'max_iters') * 0.75)
            swa_step_max_iters = (self.configer.get('solver', 'max_iters') - normal_max_iters) // 5 + 1

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.configer.get('epoch'))

        for i, data_dict in enumerate(self.train_loader):
            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))


            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(
                    self.configer.get('iters'),
                    self.scheduler, self.optimizer, backbone_list=[0,]
                )

            (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)
            self.data_time.update(time.time() - start_time)

            foward_start_time = time.time()

            if self.configer.get("network","interp"):
                kwargs = {"interp_ratio" : self.configer.get('iters') * 1.0/self.configer.get('solver', 'max_iters')}
            else:
                kwargs = {}

            outputs = self.seg_net(*inputs,**kwargs)
            self.foward_time.update(time.time() - foward_start_time)

            if self.configer.get("teacher_interp") > 0:
                with torch.no_grad():
                    teach_outputs = self.teach(*inputs)
            else:
                teach_outputs = None

            loss_start_time = time.time()

            if is_distributed():
                import torch.distributed as dist
                def reduce_tensor(inp):
                    """
                    Reduce the loss from all processes so that
                    process with rank 0 has the averaged results.
                    """
                    world_size = get_world_size()
                    if world_size < 2:
                        return inp
                    with torch.no_grad():
                        reduced_inp = inp
                        dist.reduce(reduced_inp, dst=0)
                    return reduced_inp
                loss = self.pixel_loss(outputs, targets)
                backward_loss = loss
                display_loss = reduce_tensor(backward_loss) / get_world_size()
            else:
                backward_loss = display_loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'),\
                                                                    teach_outputs=NoSplit(teach_outputs))

            self.train_losses.update(display_loss.item(), batch_size)
            self.loss_time.update(time.time() - loss_start_time)

            backward_start_time = time.time()
            self.optimizer.zero_grad()
            backward_loss.backward()
            self.optimizer.step()
            self.backward_time.update(time.time() - backward_start_time)

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0 and \
                (not is_distributed() or get_rank() == 0):
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Forward Time {foward_time.sum:.3f}s / {2}iters, ({foward_time.avg:.3f})\t'
                         'Backward Time {backward_time.sum:.3f}s / {2}iters, ({backward_time.avg:.3f})\t'
                         'Loss Time {loss_time.sum:.3f}s / {2}iters, ({loss_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                         self.configer.get('epoch'), self.configer.get('iters'),
                         self.configer.get('solver', 'display_iter'),
                         self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                         foward_time=self.foward_time, backward_time=self.backward_time, loss_time=self.loss_time,
                         data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.foward_time.reset()
                self.backward_time.reset()
                self.loss_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            # save checkpoints for swa
            if 'swa' in self.configer.get('lr', 'lr_policy') and \
               self.configer.get('iters') > normal_max_iters and \
               ((self.configer.get('iters') - normal_max_iters) % swa_step_max_iters == 0 or \
                self.configer.get('iters') == self.configer.get('solver', 'max_iters')):
                self.optimizer.update_swa()

            if self.configer.get('iters') % 200 == 0:
                Log.info("{},{}".format(self.configer.get('iters'),self.configer.get('solver', 'max_iters')))

            if self.configer.get('iters') >= self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            # if self.configer.get('epoch') % self.configer.get('solver', 'test_interval') == 0:
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0 and (not self.configer.get("val_on_test")):
                miou = self.__val()
                trial.report(miou,self.configer.get('epoch'))

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None,retSimMap=False,printallIoU=False,endEval=False,gradcam=False):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        self.pixel_loss.eval()
        start_time = time.time()
        replicas = self.evaluator.prepare_validaton()

        if gradcam:
            self.seg_net.stage4 = self.seg_net.module.backbone.stage4
            netDic = {"type":"hrnet","layer_name":"stage4","arch":self.seg_net}
            gradcam = GradCAM(netDic)
            gradcampp = GradCAMpp(netDic)
            self.seg_net.features = self.seg_net.module.backbone
            guided = GuidedBackprop(self.seg_net)

        model_id = self.configer.get("checkpoints","checkpoints_name")

        if self.configer.get("network","interp"):
            if endEval:
                ratio = 1
            else:
                ratio = self.configer.get('iters') * 1.0/self.configer.get('solver', 'max_iters')
            Log.info("Interpolation ratio : {}".format(ratio))
        else:
            ratio = 1

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            if j % 10 == 0:
                Log.info('{} images processed\n'.format(j))

            if self.configer.get('dataset') == 'lip':
                (inputs, targets, inputs_rev, targets_rev), batch_size = self.data_helper.prepare_data(data_dict, want_reverse=True)
            else:
                (inputs, targets), batch_size = self.data_helper.prepare_data(data_dict)

            if self.configer.get('dataset') == 'lip':
                inputs = torch.cat([inputs[0], inputs_rev[0]], dim=0)
                outputs = self.seg_net(inputs)
                outputs_ = self.module_runner.gather(outputs)
                if isinstance(outputs_, (list, tuple)):
                    outputs_ = outputs_[-1]
                outputs = outputs_[0:int(outputs_.size(0)/2),:,:,:].clone()
                outputs_rev = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),:,:,:].clone()
                if outputs_rev.shape[1] == 20:
                    outputs_rev[:,14,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),15,:,:]
                    outputs_rev[:,15,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),14,:,:]
                    outputs_rev[:,16,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),17,:,:]
                    outputs_rev[:,17,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),16,:,:]
                    outputs_rev[:,18,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),19,:,:]
                    outputs_rev[:,19,:,:] = outputs_[int(outputs_.size(0)/2):int(outputs_.size(0)),18,:,:]
                outputs_rev = torch.flip(outputs_rev, [3])
                outputs = (outputs + outputs_rev) / 2.
                self.evaluator.update_score(outputs, data_dict['meta'])

            elif self.data_helper.conditions.diverse_size:
                outputs = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)

                for i in range(len(outputs)):
                    loss = self.pixel_loss(outputs[i], targets[i])
                    self.val_losses.update(loss.item(), 1)
                    outputs_i = outputs[i]
                    if isinstance(outputs_i, torch.Tensor):
                        outputs_i = [outputs_i]
                    self.evaluator.update_score(outputs_i, data_dict['meta'][i:i+1])
            else:

                with torch.no_grad():
                    kwargs = {"retSimMap":retSimMap,"interp_ratio":ratio}
                    outputs = self.seg_net(*inputs,**kwargs)

                if retSimMap:
                    if j % 5 == 0:
                        pred = torch.cat([out[1].cpu() for out in outputs],dim=0)
                        attMaps = torch.cat([out[2].cpu() for out in outputs],dim=0)
                        norm = torch.cat([out[3].cpu() for out in outputs],dim=0)

                        Log.info("Saving att maps at batch {} of validation".format(j))
                        torch.save(attMaps,"results/cityscapes/simMaps{}_model{}.pt".format(j,model_id))
                        torch.save(pred,"results/cityscapes/pred{}_model{}.pt".format(j,model_id))
                        torch.save(norm,"results/cityscapes/norm{}_model{}.pt".format(j,model_id))

                        ori_img = [torch.tensor(np.array(dic["ori_img"])[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0) for dic in data_dict["meta"]]
                        ori_img = torch.cat(ori_img,dim=0)
                        torch.save(ori_img,"results/cityscapes/img{}.pt".format(j))

                        if gradcam:
                            allMask,allMask_pp,allMaps,allLab = [],[],[],[]
                            inp = inputs[0]
                            inp = torch.nn.functional.interpolate(inp,size=(768,1536))

                            for i in range(len(inp)):
                                print(inp[i:i+1].shape)
                                mask,label = gradcam(inp[i:i+1])
                                allMask.append(mask)
                                #allMask_pp.append(gradcampp(inp[i:i+1],label))
                                #allMaps.append(guided.generate_gradients(inp[i:i+1],label))
                                allLab.append(label.unsqueeze(0))
                            allMask = torch.cat(allMask,dim=0)
                            #allMask_pp = torch.cat(allMask_pp,dim=0)
                            #allMaps = torch.cat(allMaps,dim=0)
                            allLab = torch.cat(allLab,dim=0)
                            torch.save(allMask,"results/cityscapes/gradcam{}_model{}.pt".format(j,model_id))
                            #torch.save(allMask_pp,"results/cityscapes/gradcam_pp{}_model{}.pt".format(j,model_id))
                            #torch.save(allMaps,"results/cityscapes/guided{}_model{}.pt".format(j,model_id))
                            torch.save(allLab,"results/cityscapes/chosenlab{}_model{}.pt".format(j,model_id))

                    outputs = [out[:2] for out in outputs]

                with torch.no_grad():
                    try:
                        loss = self.pixel_loss(
                            outputs, targets,
                            gathered=self.configer.get('network', 'gathered')
                        )
                    except AssertionError as e:
                        print(len(outputs), len(targets))


                if not is_distributed():
                    outputs = self.module_runner.gather(outputs)
                self.val_losses.update(loss.item(), batch_size)
                self.evaluator.update_score(outputs, data_dict['meta'])

            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.evaluator.update_performance()

        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')
        cudnn.benchmark = True

        # Print the log info & reset the states.
        if not is_distributed() or get_rank() == 0:
            Log.info(
                'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                'Loss {loss.avg:.8f}\n'.format(
                    batch_time=self.batch_time, loss=self.val_losses))
            miou,allIoU = self.evaluator.print_scores()

            if printallIoU:
                with open("results/cityscapes/perf.csv","a") as text_file:
                    model_id = self.configer.get("checkpoints","checkpoints_name")
                    trial = self.configer.get("trial_nb")
                    iteration = self.configer.get('iters')
                    print("{},{},{},{}".format(model_id,trial,iteration,",".join(allIoU.astype("str"))),file=text_file)

        self.batch_time.reset()
        self.val_losses.reset()
        self.evaluator.reset()
        self.seg_net.train()
        self.pixel_loss.train()
        return miou

    def train(self,trial):
        cudnn.benchmark = True
        #self.__val(retSimMap=True,endEval=True,gradcam=True)
        #if self.configer.get('network', 'resume') is not None:
        #    if self.configer.get('network', 'resume_val'):
        #        self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #        return
        #    elif self.configer.get('network', 'resume_train'):
        #        self.__val(data_loader=self.data_loader.get_valloader(dataset='train'))
        #        return
            # return

        #if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
        #    self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))
        #    return

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train(trial)

        # use swa to average the model
        if 'swa' in self.configer.get('lr', 'lr_policy'):
            self.optimizer.swap_swa_sgd()
            self.optimizer.bn_update(self.train_loader, self.seg_net)

        if not self.configer.get("val_on_test"):
            self.__val(data_loader=self.data_loader.get_valloader(dataset='val'))

        return self.configer.get("performance")

    def val(self, data_loader=None,retSimMap=False,printallIoU=False,endEval=True,gradcam=True):
        return self.__val(data_loader,retSimMap,printallIoU=printallIoU,endEval=endEval,gradcam=gradcam)

    def summary(self):
        from lib.utils.summary import get_model_summary
        import torch.nn.functional as F
        self.seg_net.eval()

        for j, data_dict in enumerate(self.train_loader):
            print(get_model_summary(self.seg_net, data_dict['img'][0:1]))
            return


if __name__ == "__main__":
    pass
