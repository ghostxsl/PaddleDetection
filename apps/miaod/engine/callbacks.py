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
import datetime
import paddle.distributed as dist

from .checkpoint import save_model

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Callback', 'ComposeCallback', 'LogPrinter', 'Checkpointer']


class Callback(object):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_begin(self, status):
        pass

    def on_step_end(self, status):
        pass

    def on_epoch_begin(self, status):
        pass

    def on_epoch_end(self, status):
        pass

    def on_stage_begin(self, status):
        pass

    def on_stage_end(self, status):
        pass

    def on_cycle_begin(self, status):
        pass

    def on_cycle_end(self, status):
        pass

    def on_train_begin(self, status):
        pass

    def on_train_end(self, status):
        pass


class ComposeCallback(object):
    def __init__(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(
                c, Callback), "callback should be subclass of Callback"
        self._callbacks = callbacks

    def on_step_begin(self, status):
        for c in self._callbacks:
            c.on_step_begin(status)

    def on_step_end(self, status):
        for c in self._callbacks:
            c.on_step_end(status)

    def on_epoch_begin(self, status):
        for c in self._callbacks:
            c.on_epoch_begin(status)

    def on_epoch_end(self, status):
        for c in self._callbacks:
            c.on_epoch_end(status)

    def on_stage_begin(self, status):
        for c in self._callbacks:
            c.on_stage_begin(status)

    def on_stage_end(self, status):
        for c in self._callbacks:
            c.on_stage_end(status)

    def on_cycle_begin(self, status):
        for c in self._callbacks:
            c.on_cycle_begin(status)

    def on_cycle_end(self, status):
        for c in self._callbacks:
            c.on_cycle_end(status)

    def on_train_begin(self, status):
        for c in self._callbacks:
            c.on_train_begin(status)

    def on_train_end(self, status):
        for c in self._callbacks:
            c.on_train_end(status)


class LogPrinter(Callback):
    def __init__(self, trainer):
        super(LogPrinter, self).__init__(trainer)

    def on_step_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if status['mode'] == 'train' and status[
                    'step_id'] % self.trainer.cfg.log_iter == 0:
                steps_per_epoch = status['steps_per_epoch']
                batch_time = status['batch_time']
                data_time = status['data_time']
                eta_epoch = self.trainer.end_epoch - status['epoch_id']
                batch_size = self.trainer.cfg['TrainReader']['batch_size']

                space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
                eta_steps = eta_epoch * steps_per_epoch - status['step_id']
                eta_sec = eta_steps * batch_time.global_avg
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                ips = float(batch_size) / batch_time.avg
                fmt = ' '.join([
                    'Epoch: [{}-{}-{}]',
                    '[{' + space_fmt + '}/{}]',
                    'learning_rate: {lr:.6f}',
                    '{meters}',
                    'eta: {eta}',
                    'batch_cost: {btime}',
                    'data_cost: {dtime}',
                    'ips: {ips:.4f} images/s',
                ])
                fmt = fmt.format(
                    status['cycle_id'],
                    status['stage_id'],
                    status['epoch_id'],
                    status['step_id'],
                    steps_per_epoch,
                    lr=status['learning_rate'],
                    meters=status['training_staus'].log(),
                    eta=eta_str,
                    btime=str(batch_time),
                    dtime=str(data_time),
                    ips=ips)
                logger.info(fmt)
            if status['mode'] == 'eval':
                step_id = status['step_id']
                if step_id % 100 == 0:
                    logger.info("Eval iter: {}".format(step_id))

    def on_epoch_end(self, status):
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            mode = status['mode']
            if mode == 'eval':
                sample_num = status['sample_num']
                cost_time = status['cost_time']
                logger.info('Total sample number: {}, averge FPS: {}'.format(
                    sample_num, sample_num / cost_time))


class Checkpointer(Callback):
    def __init__(self, trainer):
        super(Checkpointer, self).__init__(trainer)
        self.best_ap = 0.
        self.save_dir = os.path.join(self.trainer.cfg.save_dir,
                                     self.trainer.cfg.filename)
        if hasattr(self.trainer.model, 'student_model'):
            self.weight = self.trainer.model.student_model
        else:
            self.weight = self.trainer.model

    def on_epoch_end(self, status):
        # Checkpointer only performed during training
        mode = status['mode']
        weight = None
        save_name = None
        save_best_model = status.get('save_best_model', False)
        if dist.get_world_size() < 2 or dist.get_rank() == 0:
            if mode == 'train':
                total_epoch_id = status['total_epoch_id']
                if (total_epoch_id + 1) % self.trainer.cfg.snapshot_epoch == 0 \
                        or total_epoch_id == self.trainer.total_epoch - 1:
                    save_name = str(total_epoch_id) \
                        if total_epoch_id < self.trainer.total_epoch else "model_final"
                    weight = self.weight.state_dict()
            elif mode == 'eval' and save_best_model:
                for metric in self.trainer._metrics:
                    map_res = metric.get_results()
                    if 'bbox' in map_res:
                        key = 'bbox'
                    elif 'keypoint' in map_res:
                        key = 'keypoint'
                    else:
                        key = 'mask'
                    if key not in map_res:
                        logger.warning("Evaluation results empty, this may be due to " \
                                    "training iterations being too few or not " \
                                    "loading the correct weights.")
                        return
                    if map_res[key][0] > self.best_ap:
                        self.best_ap = map_res[key][0]
                        save_name = 'best_model'
                        weight = self.weight.state_dict()
                    logger.info("Best test {} ap is {:0.3f}.".format(
                        key, self.best_ap))
            if weight and status['mode'] == 'eval':
                if self.trainer.ema_model is not None:
                    # save trainer and ema_model
                    save_model(
                        status['weight'], [
                            self.trainer.optimizer,
                            self.trainer.optimizer_scheduler
                        ],
                        self.save_dir,
                        save_name,
                        status['total_epoch_id'],
                        ema_model=weight)
                else:
                    save_model(weight, [
                        self.trainer.optimizer, self.trainer.optimizer_scheduler
                    ], self.save_dir, save_name, status['total_epoch_id'])
