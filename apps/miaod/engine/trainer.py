# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import copy
import time
from tqdm import tqdm

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA

from ppdet.core.workspace import create
from .checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import COCOMetric, VOCMetric, get_infer_results
from ppdet.data.source.sniper_coco import SniperCOCODataSet
from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from .utils import init_active_dataset_miaod, update_active_dataset_miaod
from ppdet.modeling.ops import paddle_distributed_is_initialized

from .callbacks import ComposeCallback, LogPrinter, Checkpointer
from ppdet.engine.export_utils import _dump_infer_config, _prune_input_spec

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train', validate=False):
        assert mode.lower() in ['train', 'eval', 'test'], \
            "mode should be 'train', 'eval' or 'test'"
        self.cfg = cfg
        self.mode = mode.lower()
        self.optimizer = None
        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.validate = validate

        # build model
        self.model = create(cfg.architecture)
        # normalize params for deploy
        self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        if self.mode == 'train':
            self.initial_sample_ratio = self.cfg.get('initial_sample_ratio',
                                                     0.05)
            self.delta_sample_ratio = self.cfg.get('delta_sample_ratio', 0.025)
            # build labeled and unlabeled dataset in train mode
            self._init_train_dataset()
            # build optimizer in train mode
            self.optimizer = create('OptimizerBuilder')(
                self.cfg['LearningRate']['base_lr'], self.model)
            self.optimizer_scheduler = self.build_scheduler_optimizer(
                len(self.train_labeled_loader), self.model)
            # build EMA in train mode
            self.ema_model = ModelEMA(
                self.model, decay=self.cfg.get('ema_decay', 0.9998),
                ema_decay_type=self.cfg.get('ema_decay_type', 'threshold'),
                cycle_epoch=self.cfg.get('cycle_epoch', -1)) \
                if self.cfg.get('use_ema', False) else None
            if self.validate:
                # build eval loader in train mode
                # evaluate in single device
                self.eval_dataset = self.cfg['EvalDataset']
                self.eval_batch_sampler = paddle.io.BatchSampler(
                    self.eval_dataset,
                    batch_size=self.cfg.EvalReader['batch_size'])
                if cfg.metric == 'VOC':
                    cfg['EvalReader']['collate_batch'] = False
                self.eval_loader = create('EvalReader')(
                    self.eval_dataset, cfg.worker_num, self.eval_batch_sampler)
        # EvalDataset build with BatchSampler to evaluate in single device
        elif self.mode == 'eval':
            self.eval_dataset = self.cfg['EvalDataset']
            self.eval_batch_sampler = paddle.io.BatchSampler(
                self.eval_dataset, batch_size=self.cfg.EvalReader['batch_size'])
            # If metric is VOC, need to be set collate_batch=False.
            if cfg.metric == 'VOC':
                cfg['EvalReader']['collate_batch'] = False
            self.eval_loader = create('EvalReader')(
                self.eval_dataset, cfg.worker_num, self.eval_batch_sampler)
        else:
            self.test_dataset = self.cfg['TestDataset']

        self._init_status()
        # initial default callbacks
        self._init_callbacks()
        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_train_dataset(self):
        train_dataset = self.cfg['TrainDataset']
        self.all_data_id = list(train_dataset.all_data_dict.keys())
        self.train_labeled_dataset = copy.deepcopy(train_dataset)
        self.train_unlabeled_dataset = copy.deepcopy(train_dataset)
        self.all_unlabeled_dataset = copy.deepcopy(train_dataset)

        labeled_id_list, unlabeled_id_list, all_unlabeled_id_list = \
            init_active_dataset_miaod(copy.deepcopy(self.all_data_id),
                                      self.initial_sample_ratio)

        if self._nranks > 1 and paddle_distributed_is_initialized():
            labeled_id_list = paddle.to_tensor(labeled_id_list)
            unlabeled_id_list = paddle.to_tensor(unlabeled_id_list)
            all_unlabeled_id_list = paddle.to_tensor(all_unlabeled_id_list)

            dist.broadcast(labeled_id_list, 0)
            dist.broadcast(unlabeled_id_list, 0)
            dist.broadcast(all_unlabeled_id_list, 0)

            labeled_id_list, unlabeled_id_list, all_unlabeled_id_list = \
                labeled_id_list.tolist(), unlabeled_id_list.tolist(), \
                all_unlabeled_id_list.tolist()

        self.update_dataloader(labeled_id_list, unlabeled_id_list,
                               all_unlabeled_id_list)

    def update_dataloader(self, labeled_id_list, unlabeled_id_list,
                          all_unlabeled_id_list):
        self.train_labeled_dataset.update_dataset(labeled_id_list)
        self.train_unlabeled_dataset.update_dataset(unlabeled_id_list)
        self.all_unlabeled_dataset.update_dataset(all_unlabeled_id_list)

        self.train_labeled_loader = create('TrainReader')(
            self.train_labeled_dataset, self.cfg.worker_num)
        self.train_unlabeled_loader = create('TrainReader')(
            self.train_unlabeled_dataset, self.cfg.worker_num).dataloader
        if self.cfg.metric == 'VOC':
            self.cfg['EvalReader']['collate_batch'] = False
        self.all_unlabeled_loader = create('EvalReader')(
            self.all_unlabeled_dataset, self.cfg.worker_num)

    def _init_status(self):
        self.status = {'mode': self.mode}
        if self.mode == 'train':
            self.start_cycle = 0
            self.end_cycle = self.cfg.get('cycle', 7)

            self.start_round = 0
            self.end_round = self.cfg.get('round', 2)

            # MI-AOD has 3 stages of training, which is `Label set training`,
            # `Re-weighting and minimizing instance uncertainty` and
            # `Re-weighting and maximizing instance uncertainty`.
            self.start_stage = 0
            self.stage_ratio = self.cfg.get('stage_ratio', [6, 2, 2])
            assert len(self.stage_ratio) == 3, \
                f"The length of `stage_ratio` needs to be equal to 3," \
                f" but received {len(self.stage_ratio)}. "

            self.start_epoch = 0
            self.end_epoch = self.stage_ratio[0]

            self.status['total_epoch_id'] = 0
            self.round_epoch = sum(self.stage_ratio)
            self.cycle_epoch = self.round_epoch * self.end_round + self.stage_ratio[
                0]
            self.total_epoch = self.cycle_epoch * self.end_cycle

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)

    def _init_metrics(self):
        if self.mode == 'test' or (self.mode == 'train' and not self.validate):
            self._metrics = None
            return
        classwise = self.cfg.get('classwise', False)
        if self.cfg.metric == 'COCO':
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                                if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()
            dataset = self.dataset
            if self.mode == 'train' and self.validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
        elif self.cfg.metric == 'VOC':
            self._metrics = [
                VOCMetric(
                    label_list=self.eval_dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = None

    def _reset_metrics(self):
        if self._metrics is None:
            return
        for metric in self._metrics:
            metric.reset()

    def load_pretrain_weights(self, weights):
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights)
        logger.debug("Load pretrain weights {} to start training".format(
            weights))

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            total_epoch_id = load_weight(
                self.model.student_model, weights,
                [self.optimizer, self.optimizer_scheduler])
        else:
            total_epoch_id = load_weight(
                self.model, weights,
                [self.optimizer, self.optimizer_scheduler], self.ema_model)
        self.start_cycle, self.start_round, self.start_stage, self.start_epoch\
            = self.total_epoch_id2status_id(total_epoch_id)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def build_scheduler_optimizer(self, steps_per_epoch, model):
        lr_scheduler = create('LearningRate')(steps_per_epoch)
        return create('OptimizerBuilder')(lr_scheduler, model)

    def total_epoch_id2status_id(self, total_epoch_id):
        cycly_id = total_epoch_id // self.cycle_epoch
        epochs = total_epoch_id % self.cycle_epoch
        round_id = epochs // self.round_epoch
        epochs = epochs % self.round_epoch
        if round_id == self.end_round:
            stage_id = len(self.stage_ratio)
        else:
            for i in range(len(self.stage_ratio)):
                if epochs >= self.stage_ratio[i]:
                    epochs -= self.stage_ratio[i]
                else:
                    stage_id = i
                    break
        epoch_id = epochs
        return cycly_id, round_id, stage_id, epoch_id

    def train(self):
        assert self.mode == 'train', "Model not in 'train' mode"
        model = self.model

        # convert sync_bn to model
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
                   self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # enable auto mixed precision mode
        if self.use_amp:
            self.scaler = paddle.amp.GradScaler(
                enable=self.cfg.use_gpu,
                init_loss_scaling=self.cfg.get('init_loss_scaling', 1024))
            model = paddle.amp.decorate(models=model, level=self.amp_level)

        # build distributed model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
            self.optimizer_scheduler = \
                fleet.distributed_optimizer(self.optimizer_scheduler)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)

        # initial training status
        self.status.update({
            'cycle_id': self.start_cycle,
            'round_id': self.start_round,
            'stage_id': self.start_stage,
            'epoch_id': self.start_epoch,
            'step_id': 0,
        })
        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        # start training
        self._compose_callback.on_train_begin(self.status)
        for cycle_id in range(self.start_cycle, self.end_cycle):
            self.status['mode'] = 'train'
            self.status['cycle_id'] = cycle_id
            self.status['steps_per_epoch'] = len(self.train_labeled_loader)
            self.run_one_cycle(model)

            # Informative Image Selection
            self.informative_image_select(model, self.all_unlabeled_loader)
            # build next cycle scheduler optimizer
            self.optimizer_scheduler = self.build_scheduler_optimizer(
                len(self.train_labeled_loader), self.model)
            if self.cfg.get('fleet', False):
                self.optimizer_scheduler = \
                    fleet.distributed_optimizer(self.optimizer_scheduler)
        self._compose_callback.on_train_end(self.status)

    def run_one_cycle(self, model):
        self._compose_callback.on_cycle_begin(self.status)
        for round_id in range(self.start_round, self.end_round):
            self.status['round_id'] = round_id
            for stage_id in range(self.start_stage, len(self.stage_ratio)):
                self.status['stage_id'] = stage_id
                self.end_epoch = self.stage_ratio[stage_id]
                self.run_one_stage(model, stage_id, self.optimizer)
            self.start_stage = 0
        self.start_round = 0
        # run last one `Label set training` stage in each cycle
        self.status['stage_id'] = 3
        self.end_epoch = self.stage_ratio[0]
        self.run_one_stage(model, 0, self.optimizer_scheduler, self.validate)
        self._compose_callback.on_cycle_end(self.status)

    def run_one_stage(self, model, stage_id, optimizer, validate=False):
        assert stage_id < len(self.stage_ratio), \
            f"MI-AOD has {len(self.stage_ratio)} stages," \
            f" but received `stage_id`: {stage_id}. "
        self._compose_callback.on_stage_begin(self.status)
        for epoch_id in range(self.start_epoch, self.end_epoch):
            self.status['epoch_id'] = epoch_id
            self.train_labeled_dataset.set_epoch(epoch_id)
            self.train_unlabeled_dataset.set_epoch(epoch_id)
            self.run_one_epoch(model, stage_id, optimizer)

            # eval in train
            is_snapshot = (self._nranks < 2 or self._local_rank == 0) and \
                          ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or
                           epoch_id + 1 == self.end_epoch)
            if validate and is_snapshot:
                self._eval_in_train()
        self._compose_callback.on_stage_end(self.status)
        self.start_epoch = 0

    def run_one_epoch(self, model, stage_id, optimizer):
        model.train()
        self.status['mode'] = 'train'
        self._compose_callback.on_epoch_begin(self.status)
        if stage_id != 0:
            train_unlabeled_loader = iter(self.train_unlabeled_loader)
        iter_tic = time.time()
        for step_id, data in enumerate(self.train_labeled_loader):
            self.status['data_time'].update(time.time() - iter_tic)
            self.status['step_id'] = step_id
            data['epoch_id'] = self.status['epoch_id']
            data['stage_id'] = stage_id
            data['is_unlabeled'] = False
            self._compose_callback.on_step_begin(self.status)
            # labeled image training
            self.run_one_step(model, data, optimizer)
            if stage_id != 0:
                # unlabeled image training
                data = next(train_unlabeled_loader)
                data['epoch_id'] = self.status['epoch_id']
                data['stage_id'] = stage_id
                data['is_unlabeled'] = True
                self.run_one_step(model, data, optimizer)
            self.status['batch_time'].update(time.time() - iter_tic)
            iter_tic = time.time()
            self._compose_callback.on_step_end(self.status)
        if stage_id != 0:
            try:
                next(train_unlabeled_loader)
            except StopIteration:
                train_unlabeled_loader._try_shutdown_all()
                del train_unlabeled_loader
        self.status['total_epoch_id'] += 1
        self._compose_callback.on_epoch_end(self.status)

    def run_one_step(self, model, data, optimizer):
        if self.use_amp:
            with paddle.amp.auto_cast(
                    enable=self.cfg.use_gpu, level=self.amp_level):
                # model forward
                outputs = model(data)
                loss = outputs['loss']
            # model backward
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            # in dygraph mode, optimizer.minimize is equal to optimizer.step
            self.scaler.minimize(optimizer, scaled_loss)
        else:
            # model forward
            outputs = model(data)
            loss = outputs['loss']
            # model backward
            loss.backward()
            optimizer.step()
        curr_lr = optimizer.get_lr()
        if isinstance(optimizer._learning_rate,
                      paddle.optimizer.lr.LRScheduler):
            optimizer._learning_rate.step()
        optimizer.clear_grad()
        self.status['learning_rate'] = curr_lr

        if self._nranks < 2 or self._local_rank == 0:
            self.status['training_staus'].update(outputs)
        if self.ema_model is not None:
            self.ema_model.update()

    def informative_image_select(self, model, loader):
        assert len(loader.dataset) > 0, \
            'There is no remaining unlabeled data to compute uncertainty.'
        # prepare tensor to broadcast
        num_delta_sample = int(
            len(self.train_labeled_dataset.all_data_dict) *
            self.delta_sample_ratio)
        assert num_delta_sample <= len(loader.dataset)
        labeled_id = paddle.zeros(
            [len(self.train_labeled_dataset) + num_delta_sample], dtype='int64')
        unlabeled_id = paddle.zeros_like(labeled_id)
        if len(loader.dataset) - num_delta_sample > 0:
            num_all_unlabeled = len(loader.dataset) - num_delta_sample
        else:
            num_all_unlabeled = 0
        all_unlabeled_id = paddle.zeros([num_all_unlabeled], dtype='int64')

        # Informative Image Selection
        # use best_model to select informative image
        best_model_path = os.path.join(self._callbacks[1].save_dir,
                                       "best_model.pdparams")
        if os.path.exists(best_model_path):
            src_weight = copy.deepcopy(model.state_dict())
            model.set_state_dict(paddle.load(best_model_path))
            logger.info('Using weights: {} to compute uncertainty.'.format(
                best_model_path))

        uncertainty = self._eval_uncertainty(model, loader)

        if self._nranks > 1 and paddle_distributed_is_initialized():
            dist.barrier()

        if os.path.exists(best_model_path):
            model.set_state_dict(src_weight)

        if self._nranks < 2 or self._local_rank == 0:
            labeled_id, unlabeled_id, all_unlabeled_id = \
                update_active_dataset_miaod(uncertainty,
                self.train_labeled_dataset.data_id_list,
                self.all_unlabeled_dataset.data_id_list,
                self.delta_sample_ratio)
            labeled_id, unlabeled_id, all_unlabeled_id = \
                paddle.to_tensor(labeled_id), paddle.to_tensor(unlabeled_id),\
                paddle.to_tensor(all_unlabeled_id)

        if self._nranks > 1 and paddle_distributed_is_initialized():
            dist.barrier()
            dist.broadcast(labeled_id, 0)
            dist.broadcast(unlabeled_id, 0)
            dist.broadcast(all_unlabeled_id, 0)

        self.update_dataloader(labeled_id.tolist(),
                               unlabeled_id.tolist(), all_unlabeled_id.tolist())

    def _eval_uncertainty(self, model, loader):
        # eval image uncertainty
        model.eval()
        self.status['mode'] = 'eval'
        uncertainty_list = []
        with paddle.no_grad():
            self._compose_callback.on_epoch_begin(self.status)
            tic = time.time()
            for step_id, data in enumerate(loader):
                self.status['step_id'] = step_id
                self._compose_callback.on_step_begin(self.status)
                # forward
                data['eval_uncertainty'] = True
                outs = model(data)
                uncertainty = self._dist_all_gather(outs['uncertainty'])
                uncertainty_list.append(uncertainty.numpy())
                self._compose_callback.on_step_end(self.status)

            uncertainty_list = np.concatenate(uncertainty_list, axis=0)
            self.status['sample_num'] = len(uncertainty_list)
            self.status['cost_time'] = time.time() - tic
            self._compose_callback.on_epoch_end(self.status)
        return uncertainty_list[:len(loader.dataset)]

    def _eval_in_train(self):
        self._reset_metrics()
        if self.ema_model is not None:
            # apply ema weight on model
            self.status['weight'] = copy.deepcopy(self.model.state_dict())
            self.model.set_dict(self.ema_model.apply())

        with paddle.no_grad():
            self.status['save_best_model'] = True
            self._eval_with_loader(self.eval_loader)

        if self.ema_model is not None:
            # reset original weight
            self.model.set_dict(self.status['weight'])
            self.status.pop('weight')

    def _eval_with_loader(self, loader):
        self.status['mode'] = 'eval'
        self.model.eval()
        model = self.model
        if self.cfg.get('print_flops', False):
            flops_loader = create('EvalReader')(
                self.eval_dataset, self.cfg.worker_num, self.eval_batch_sampler)
            self._flops(flops_loader)
        self._compose_callback.on_epoch_begin(self.status)
        tic = time.time()
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            outs = model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = len(self.eval_dataset)
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        with paddle.no_grad():
            self._eval_with_loader(self.eval_loader)

    def predict_uncertainty(self, images, output_dir='output'):
        self.test_dataset.set_images(images)
        loader = create('TestReader')(self.test_dataset, self.cfg.worker_num)
        imid2path = self.test_dataset.get_imid2path()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.test_dataset, 0)
            self._flops(flops_loader)

        # build distributed model
        model = self.model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
        elif self._nranks > 1:
            model = paddle.DataParallel(model)
        # Run Infer
        uncertainty = self._eval_uncertainty(model, loader)
        return uncertainty, imid2path

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_txt=False):
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)
        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for outs in results:
            batch_res = get_infer_results(outs, clsid2catid)
            bbox_num = outs['bbox_num']

            start = 0
            for i, im_id in enumerate(outs['im_id']):
                image_path = imid2path[int(im_id)]
                image = Image.open(image_path).convert('RGB')
                image = ImageOps.exif_transpose(image)
                self.status['original_image'] = np.array(image.copy())

                end = start + bbox_num[i]
                bbox_res = batch_res['bbox'][start:end] \
                        if 'bbox' in batch_res else None
                mask_res = batch_res['mask'][start:end] \
                        if 'mask' in batch_res else None
                segm_res = batch_res['segm'][start:end] \
                        if 'segm' in batch_res else None
                keypoint_res = batch_res['keypoint'][start:end] \
                        if 'keypoint' in batch_res else None
                image = visualize_results(
                    image, bbox_res, mask_res, segm_res, keypoint_res,
                    int(im_id), catid2name, draw_threshold)
                self.status['result_image'] = np.array(image.copy())
                if self._compose_callback:
                    self._compose_callback.on_step_end(self.status)
                # save image with detection
                save_name = self._get_save_image_name(output_dir, image_path)
                logger.info("Detection bbox results save in {}".format(
                    save_name))
                image.save(save_name, quality=95)
                if save_txt:
                    save_path = os.path.splitext(save_name)[0] + '.txt'
                    results = {}
                    results["im_id"] = im_id
                    if bbox_res:
                        results["bbox_res"] = bbox_res
                    if keypoint_res:
                        results["keypoint_res"] = keypoint_res
                    save_result(save_path, results, catid2name, draw_threshold)
                start = end

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        return os.path.join(output_dir, "{}".format(name)) + ext

    def _get_infer_cfg_and_input_spec(self, save_dir, prune_input=True):
        image_shape = None
        im_shape = [None, 2]
        scale_factor = [None, 2]
        test_reader_name = 'TestReader'
        if 'inputs_def' in self.cfg[test_reader_name]:
            inputs_def = self.cfg[test_reader_name]['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[None, 3, -1, -1] as default
        if image_shape is None:
            image_shape = [None, 3, -1, -1]

        if len(image_shape) == 3:
            image_shape = [None] + image_shape
        else:
            im_shape = [image_shape[0], 2]
            scale_factor = [image_shape[0], 2]

        if hasattr(self.model, 'deploy'):
            self.model.deploy = True

        for layer in self.model.sublayers():
            if hasattr(layer, 'convert_to_deploy'):
                layer.convert_to_deploy()

        export_post_process = self.cfg['export'].get(
            'post_process', False) if hasattr(self.cfg, 'export') else True
        export_nms = self.cfg['export'].get('nms', False) if hasattr(
            self.cfg, 'export') else True
        export_benchmark = self.cfg['export'].get(
            'benchmark', False) if hasattr(self.cfg, 'export') else False
        if hasattr(self.model, 'fuse_norm'):
            self.model.fuse_norm = self.cfg['TestReader'].get('fuse_normalize',
                                                              False)
        if hasattr(self.model, 'export_post_process'):
            self.model.export_post_process = export_post_process if not export_benchmark else False
        if hasattr(self.model, 'export_nms'):
            self.model.export_nms = export_nms if not export_benchmark else False
        if export_post_process and not export_benchmark:
            image_shape = [None] + image_shape[1:]

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(save_dir, 'infer_cfg.yml'), image_shape,
                           self.model)

        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image'),
            "im_shape": InputSpec(
                shape=im_shape, name='im_shape'),
            "scale_factor": InputSpec(
                shape=scale_factor, name='scale_factor')
        }]
        if self.cfg.architecture == 'DeepSORT':
            input_spec[0].update({
                "crops": InputSpec(
                    shape=[None, 3, 192, 64], name='crops')
            })
        if prune_input:
            static_model = paddle.jit.to_static(
                self.model, input_spec=input_spec)
            # NOTE: dy2st do not pruned program, but jit.save will prune program
            # input spec, prune input spec here and save with pruned input spec
            pruned_input_spec = _prune_input_spec(
                input_spec, static_model.forward.main_program,
                static_model.forward.outputs)
        else:
            static_model = None
            pruned_input_spec = input_spec

        # TODO: Hard code, delete it when support prune input_spec.
        if self.cfg.architecture == 'PicoDet' and not export_post_process:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image')
            }]

        return static_model, pruned_input_spec

    def export(self, output_dir='output_inference'):
        self.model.eval()
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        static_model, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir)

        # dy2st and save model
        if 'slim' not in self.cfg or self.cfg['slim_type'] != 'QAT':
            paddle.jit.save(
                static_model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        else:
            self.cfg.slim.save_quantized_model(
                self.model,
                os.path.join(save_dir, 'model'),
                input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))

    def post_quant(self, output_dir='output_inference'):
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, data in enumerate(self.loader):
            self.model(data)
            if idx == int(self.cfg.get('quant_batch_num', 10)):
                break

        # TODO: support prune input_spec
        _, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, prune_input=False)

        self.cfg.slim.save_quantized_model(
            self.model,
            os.path.join(save_dir, 'model'),
            input_spec=pruned_input_spec)
        logger.info("Export Post-Quant model and saved in {}".format(save_dir))

    def _flops(self, loader):
        self.model.eval()
        try:
            import paddleslim
        except Exception as e:
            logger.warning(
                'Unable to calculate flops, please install paddleslim, for example: `pip install paddleslim`'
            )
            return

        from paddleslim.analysis import dygraph_flops as flops
        input_data = None
        for data in loader:
            input_data = data
            break

        input_spec = [{
            "image": input_data['image'][0].unsqueeze(0),
            "im_shape": input_data['im_shape'][0].unsqueeze(0),
            "scale_factor": input_data['scale_factor'][0].unsqueeze(0)
        }]
        flops = flops(self.model, input_spec) / (1000**3)
        logger.info(" Model FLOPs : {:.6f}G. (image shape is {})".format(
            flops, input_data['image'][0].unsqueeze(0).shape))

    def _dist_all_gather(self, tensor):
        if self._nranks < 2:
            return tensor
        tensor, size_list = self._get_pad_all_gather_tensor(tensor)
        tensor_list = []
        dist.all_gather(tensor_list, tensor)
        tensor_list = [a[:b] for a, b in zip(tensor_list, size_list)]
        return paddle.concat(tensor_list)

    def _get_pad_all_gather_tensor(self, tensor):
        # pad the tensor because paddle.distributed.all_gather does not support
        # gather tensors of different shapes
        local_size = tensor.shape[0]
        size_list = []
        dist.all_gather(size_list, paddle.to_tensor(local_size))
        size_list = [a.item() for a in size_list]
        max_size = max(size_list)
        pad_shape = tensor.shape
        pad_shape[0] = max_size - pad_shape[0]
        if local_size < max_size:
            tensor = paddle.concat(
                [tensor, paddle.zeros(
                    pad_shape, dtype=tensor.dtype)])
        return tensor, size_list
