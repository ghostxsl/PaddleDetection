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

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import load_config, merge_config
from apps.miaod.engine import Trainer
from ppdet.engine import init_parallel_env, set_random_seed, init_fleet_env

import ppdet.utils.cli as cli
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')


def parse_args():
    parser = cli.ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--enable_ce",
        type=bool,
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--fleet", action='store_true', default=False, help="Use fleet or not")
    parser.add_argument(
        '--save_prediction_only',
        action='store_true',
        default=False,
        help='Whether to save the evaluation results only')
    parser.add_argument(
        '--profiler_options',
        type=str,
        default=None,
        help="The option of profiler, which should be in "
        "format \"key1=value1;key2=value2;key3=value3\"."
        "please see ppdet/utils/profiler.py for detail.")

    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # init fleet environment
    if cfg.fleet:
        init_fleet_env(cfg.get('find_unused_parameters', False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    trainer = Trainer(cfg, mode='train', validate=FLAGS.eval)

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_pretrain_weights(cfg.pretrain_weights)

    # training
    trainer.train()


def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    cfg['amp'] = FLAGS.amp
    cfg['fleet'] = FLAGS.fleet
    cfg['save_prediction_only'] = FLAGS.save_prediction_only
    cfg['profiler_options'] = FLAGS.profiler_options
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    if 'norm_type' in cfg and cfg['norm_type'] == 'sync_bn' and not cfg.use_gpu:
        cfg['norm_type'] = 'bn'

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_npu(cfg.use_npu)
    check.check_version()

    run(FLAGS, cfg)


if __name__ == "__main__":
    main()
